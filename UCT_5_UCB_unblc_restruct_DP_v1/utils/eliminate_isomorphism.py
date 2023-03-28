import copy
import itertools
import json
from datetime import time
from time import sleep

from utils.util import get_sim_configs, mkdir, read_reward_hash, get_topology_from_hash, read_reward_hash_list
# from config import configs
from itertools import product
from ucts.GetReward import calculate_reward
from SimulatorSetPara.build_topology import nets_to_ngspice_files, convert_to_netlist
from SimulatorSetPara.simulation import simulate_topologies
from SimulatorSetPara.simulation_analysis import analysis_topologies


def get_component_priorities(preset_component='Sa'):
    # component_priorities = {'inductor': 0, 'capacitor': 1, 'FET-A': 2, 'FET-B': 3}
    if preset_component == 'Sa':
        return {'C': 3, 'L': 2, 'Sa': 0, 'Sb': 1}
    elif preset_component == 'Sb':
        return {'C': 3, 'L': 2, 'Sa': 1, 'Sb': 0}


def sort_components(set_components, topo_priority):
    result = []
    reverse_topo_priority = {}
    for k, v in topo_priority.items():
        reverse_topo_priority[v] = k
    sorted_components_count = [0 for _ in range(len(topo_priority))]
    for i in set_components:
        sorted_components_count[topo_priority[i]] += 1
    # print(sorted_components_count)
    for i in range(len(topo_priority)):
        for counts in range(sorted_components_count[i]):
            result.append(reverse_topo_priority[i])
    return tuple(result)


def unblc_comp_set_mapping(basic_components, num_components, preset_component='Sa'):
    count = 0
    set_count_mapping = {}
    set_size = num_components-1
    count_weights = {}
    topo_priority = get_component_priorities(preset_component)
    for set_n in itertools.product(basic_components, repeat=num_components):
        count += 1
        sorted_set_n = sort_components(set_n, topo_priority)
        if sorted_set_n in set_count_mapping:
            set_count_mapping[sort_components(sorted_set_n, topo_priority)] += 1
        else:
            set_count_mapping[sort_components(sorted_set_n, topo_priority)] = 1
    # for k, v in set_count_mapping.items():
    #     print(k, ' ^ ', v)
    # while set_size > -1:
    #     for set_n in itertools.product(basic_components, repeat=set_size):
    #         sorted_set_n = sort_components(set_n, topo_priority)
    #         print(sorted_set_n)
    #         if sorted_set_n in set_count_mapping:
    #             continue
    #         else:
    #             if len(sorted_set_n) > 0:
    #                 last_selected_component_type = sorted_set_n[-1]
    #             inter_count = 0
    #             branch_num = 0
    #             for i in range(len(basic_components)):
    #                 if len(sorted_set_n) == 0 or \
    #                         (topo_priority[basic_components[i]] >= topo_priority[last_selected_component_type]):
    #                     list_set_n = list(sorted_set_n)
    #                     list_set_n.append(basic_components[i])
    #                     inter_count += set_count_mapping[tuple(list_set_n)]
    #                     branch_num += 1
    #             set_count_mapping[sorted_set_n] = inter_count
    #             for i in range(len(basic_components)):
    #                 if len(sorted_set_n) == 0 or \
    #                         (topo_priority[basic_components[i]] >= topo_priority[last_selected_component_type]):
    #                     list_set_n = list(sorted_set_n)
    #                     list_set_n.append(basic_components[i])
    #                     count_weights[tuple(list_set_n)] = (inter_count / branch_num) / set_count_mapping[
    #                         tuple(list_set_n)]
    #     set_size -= 1


    count_weights[()] = 1
    # set_count_mapping[()] = len(basic_components)**num_components
    for k, v in set_count_mapping.items():
        print(k, ":", v)
    # added_count_weights = {}
    # for k, v in count_weights.items():
    #     print(k, " * ", v)
    #     list_set_n = list(k)
    #     list_set_n.insert(0, preset_component)
    #     added_count_weights[tuple(list_set_n)] = v
    #     print(tuple(list_set_n), v)
    return set_count_mapping, count_weights


def count_large_effi(history_file):
    count = 0
    fo_conf = open(history_file, "r")
    target_in_data = []
    target_with_effi = {}
    while True:
        line = fo_conf.readline()
        if not line:
            break
        if -1 != line.find("The file name:"):
            # print(line)
            target = line.split(':')[1]
            target_tmp = target
        # print(target_tmp[0:-1])

        elif line == "y:\n":
            line = fo_conf.readline()
            left_idx = line.index('[')
            right_idx = line.index(']')
            effi = line[left_idx + 1:right_idx]
            # print(effi)
            # print(float(effi))
            if float(effi) > 0.1:
                # print(float(effi))
                count += 1
                target_in_data.append(target_tmp[1:-1])
                target_with_effi[target_tmp[1:-1]] = float(effi)
    fo_conf.close()
    return target_in_data, target_with_effi


def filter_valid_edges(list_of_edge):
    edge_list = []
    for edge in list_of_edge:
        if isinstance(edge[1], int):
            edge_list.append(edge)
    # print(list_of_edge)
    # print(edge_list)
    return edge_list


def cluster_edges_for_connection_nodes(edge_list):
    clusters = {}
    add_edges = []
    for edge in edge_list:
        if clusters.__contains__(edge[1]):
            clusters[edge[1]].append(edge[0])
        else:
            clusters[edge[1]] = []
            clusters[edge[1]].append(edge[0])
    return clusters, add_edges


def get_need_add_edges(edge_list, port_2_idx):
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState
    clusters, add_edges = cluster_edges_for_connection_nodes(edge_list)
    for k, v in clusters.items():
        for i in range(len(v) - 1):
            add_edges.append(TopoGenAction('edge', [port_2_idx[v[i]], port_2_idx[v[i + 1]]]))
    return add_edges


def get_counts(component_pool):
    count_map = {"FET-A": 0, "FET-B": 0, "capacitor": 0, "inductor": 0}
    for component in component_pool:
        for i in range(len(component)):
            if component[len(component) - 1 - i] == '-':
                split_place = len(component) - 1 - i
                component_name = component[:split_place]
                count = int(component[split_place + 1:])
                if count_map[component_name] < count + 1:
                    count_map[component_name] = count + 1
                # print("-----------", component_name, count)
                break
    # print(count_map)
    return count_map


def change_key_2_int(dict):
    dict_tmp = {}
    for k, v in dict.items():
        dict_tmp[int(k)] = v
    return dict_tmp


def sort_orders(candidate_parameters):
    """
    use the candidate parameters and all possible device to generate a dict, key is the node_para, value is the priority
    if dict[n_0]<dict[n_1], n_0 should place before n_1 in formed component pool
    """
    cap_paras = candidate_parameters["capacitor"]
    ind_paras = candidate_parameters["inductor"]
    FETA_paras = candidate_parameters["FETA"]
    FETB_paras = candidate_parameters["FETB"]
    dict = {}
    component_list = []
    component_list.append("GND-1")
    component_list.append("VIN-1")
    component_list.append("VOUT-1")
    cap_paras.sort()
    for cap_para in cap_paras:
        component_list.append("capacitor-" + str(cap_para))
    ind_paras.sort()
    for ind_para in ind_paras:
        component_list.append("inductor-" + str(ind_para))
    FETA_paras.sort()
    for FETA_para in FETA_paras:
        component_list.append("FET-A-" + str(FETA_para))
    FETB_paras.sort()
    for FETB_para in FETB_paras:
        component_list.append("FET-B-" + str(FETB_para))
    for i in range(len(component_list)):
        dict[component_list[i]] = i
    return dict


def init_candidate_parameters():
    candidate_parameters = {"capacitor": [5 / 200], "inductor": [150 / 200], "FETA": [10000 / 100000],
                            "FETB": [25000 / 100000]}
    # candidate_parameters["capacitor"] = [1 / 200, 5 / 200, 20 / 200, 50 / 200]
    # candidate_parameters["inductor"] = [1 / 200, 10 / 200, 50 / 200, 150 / 200]
    return candidate_parameters


def insert_node_reorder_port_pool(component_orders, nodes_with_para,
                                  new_ordered_node_list, node_with_para, node):
    flag = None
    order = component_orders[node_with_para]
    for i in range(len(nodes_with_para)):
        if order < component_orders[nodes_with_para[i]]:
            flag = 1
            break
    if flag == 1:
        nodes_with_para.insert(i, node_with_para)
        new_ordered_node_list.insert(i, node)
    else:
        nodes_with_para.append(node_with_para)
        new_ordered_node_list.append(node)
    return nodes_with_para, new_ordered_node_list


def instance_to_isom_str(current, analytics=False):
    """
    get the formed string of a instance
    :param current: a topology instance
    :return: the stringed list of node_para form and it's edge-list
    """
    candidate_parameters = init_candidate_parameters()
    # component_orders: ordered all device node with parameter
    component_orders = sort_orders(candidate_parameters)
    list_of_node, list_of_edge, netlist, joint_list, list_of_node_2, list_of_edge_2 = \
        convert_to_netlist(current.graph,
                           current.component_pool,
                           current.port_pool,
                           current.parent,
                           current.comp2port_mapping)
    list_of_edge_tmp = []
    for edge in list_of_edge:
        list_of_edge_tmp.append([edge[0], edge[1]])
    list_of_edge = list_of_edge_tmp
    new_ordered_node_list = ["GND", "VIN", "VOUT"]
    origin_list_of_node = list_of_node
    parameters = current.parameters
    nodes_with_para = ["GND-1", "VIN-1", "VOUT-1"]
    for node in origin_list_of_node:
        if node not in ["GND", "VIN", "VOUT"] and isinstance(node, str):
            if node.find("right") == -1 and node.find("left") == -1:
                # find the deivce node and replace with "device+parameter"
                count_idx = node.rfind('-')
                node_with_para = node[:count_idx]
                if not analytics:
                    node_with_para = node_with_para + "-" + str(sum(parameters[node]))
                else:
                    node_with_para = node_with_para + "-1"
                # order the deivce node with
                nodes_with_para, new_ordered_node_list = insert_node_reorder_port_pool(component_orders,
                                                                                       nodes_with_para,
                                                                                       new_ordered_node_list,
                                                                                       node_with_para, node)

    # get the connection nodes linked device nodes
    origin_list_of_edge = list_of_edge
    edge_list = filter_valid_edges(origin_list_of_edge)
    # print(edge_list)
    # clusters: connection nodes(int) and the list of it connected nodes
    conneted_nodes_cluster, _ = cluster_edges_for_connection_nodes(edge_list)
    connection_nodes = {}
    sort_connection_nodes = []
    # k:connection nodes(int) v: list of it connected nodes
    for k, v in conneted_nodes_cluster.items():
        connected_nodes_of_k = conneted_nodes_cluster[k]
        connected_device_nodes_reordered = []
        for connected_node in connected_nodes_of_k:
            # ports change to device nodes
            connected_node = connected_node.replace("-right", "")
            connected_node = connected_node.replace("-left", "")
            # get connected_node corresponding node_para form
            connected_node_idx = new_ordered_node_list.index(connected_node)
            conneted_device_node_with_para = nodes_with_para[connected_node_idx]
            order = component_orders[conneted_device_node_with_para]
            flag = None
            for i in range(len(connected_device_nodes_reordered)):
                if order < component_orders[connected_device_nodes_reordered[i]]:
                    flag = 1
                    break
            if flag == 1:
                connected_device_nodes_reordered.insert(i, conneted_device_node_with_para)
            else:
                connected_device_nodes_reordered.append(conneted_device_node_with_para)
        # print(connected_device_nodes_reordered)
        str_connection_node = ""
        for tmp in connected_device_nodes_reordered:
            str_connection_node += tmp + "#"
        # print(str_connection_node)

        sort_connection_nodes.append(str_connection_node)
        connection_nodes[k] = str_connection_node
    # we order the connection node with alphabet order and add it to the whole node list
    sort_connection_nodes = sorted(sort_connection_nodes, key=str.lower)
    nodes_with_para = nodes_with_para + sort_connection_nodes
    print(sort_connection_nodes)
    print(nodes_with_para)
    # reorder the origin int formed connection node
    reorder_int_connection_node = []
    for k, v in conneted_nodes_cluster.items():
        print(connection_nodes[k])
        idx_tmp = sort_connection_nodes.index(connection_nodes[k])
        flag = None
        for x in range(len(reorder_int_connection_node)):
            print(reorder_int_connection_node[x])
            print(connection_nodes[reorder_int_connection_node[x]])
            print(sort_connection_nodes.index(connection_nodes[reorder_int_connection_node[x]]))
            if idx_tmp < sort_connection_nodes.index(connection_nodes[reorder_int_connection_node[x]]):
                flag = 1
                break
        if flag == 1:
            reorder_int_connection_node.insert(x, k)
        else:
            reorder_int_connection_node.append(k)
    # print(reorder_int_connection_node)
    new_ordered_node_list = new_ordered_node_list + reorder_int_connection_node
    # print(new_ordered_node_list)

    idx_new_ordered_nodes_mapping = {}
    print(new_ordered_node_list)
    for n in range(len(new_ordered_node_list)):
        idx_new_ordered_nodes_mapping[new_ordered_node_list[n]] = n
    print(idx_new_ordered_nodes_mapping)
    print()
    idxs_edge_list = []
    for edge in edge_list:
        edge[0] = edge[0].replace("-right", "")
        edge[0] = edge[0].replace("-left", "")
        idxs_edge_list.append([idx_new_ordered_nodes_mapping[edge[0]], idx_new_ordered_nodes_mapping[edge[1]]])
    print(idxs_edge_list)

    # 2-D list sorting
    # [1,3][2,1][1,0]--> [1,0][1,3][2,1]
    idxs_edge_list = sorted(idxs_edge_list, key=(lambda x: [x[0], x[1]]))
    print(idxs_edge_list)
    # print(idxs_edge_list)
    find_flag = 0
    return nodes_with_para, idxs_edge_list


def maintain_reward_hash_no_edge(no_isomorphism_reward_hash, need_insert_topo_info):
    """
    insert a topo information in the reward hash table without isomorphism
    :param no_isomorphism_reward_hash: the reward hash table without isomorphism
    :param need_insert_topo_info: list, first item is the formed node list, second item is the reward, efficiency, vout
    :return: if the topo is in no isomorphism reward hash table, do nothing and return False,
    else insert and return True
    """
    nodes_with_para, ordered_edge_list = need_insert_topo_info[0]
    efficiency_information = need_insert_topo_info[1]
    if no_isomorphism_reward_hash.__contains__(str(nodes_with_para)):
        return False
    else:
        no_isomorphism_reward_hash[str(nodes_with_para)] = efficiency_information
        return True
    return False


def maintain_reward_hash_with_edge(no_isomorphism_reward_hash, need_insert_topo_info):
    """
    insert a topo information in the reward hash table without isomorphism
    :param no_isomorphism_reward_hash: the reward hash table without isomorphism
    :param need_insert_topo_info: list, first item is the formed node list, second item is the reward, efficiency, vout
    :return: if the topo is in no isomorphism reward hash table, do nothing and return False,
    else insert and return True
    """
    nodes_with_para, ordered_edge_list = need_insert_topo_info[0]
    efficiency_information = need_insert_topo_info[1]
    topo_info = str(nodes_with_para) + "|" + str(ordered_edge_list)
    if no_isomorphism_reward_hash.__contains__(topo_info):
        return False
    else:
        no_isomorphism_reward_hash[topo_info] = efficiency_information
        return True
    return False


def eliminate_isomorphism_for_json(target_in_data, json_file):
    """
    :param target_in_data: the selected PCC-... in data_2.json
    :param json_file: the read data json file(data_2.json)
    :return: the eliminated isomorphism targets
    example: target in data can be [pcc-1,pcc-2,pcc-3],if pcc-1 and pcc-2 are isomorphism, we just return [pcc-1,pcc3]
    """
    result_targets = []
    result_target_nets = []
    candidate_parameters = init_candidate_parameters()
    # component_orders: ordered all device node with parameter
    component_orders = sort_orders(candidate_parameters)
    for target in target_in_data:
        new_ordered_node_list = ["GND", "VIN", "VOUT"]
        origin_list_of_node = json_file[target]["list_of_node"]
        parameters = json_file[target]["parameter_list"]
        nodes_with_para = ["GND-1", "VIN-1", "VOUT-1"]
        for node in origin_list_of_node:
            if node not in ["GND", "VIN", "VOUT"] and isinstance(node, str):
                if node.find("right") == -1 and node.find("left") == -1:
                    # find the deivce node and replace with "device+parameter"
                    count_idx = node.rfind('-')
                    node_with_para = node[:count_idx]
                    node_with_para = node_with_para + "-" + str(sum(parameters[node]))
                    # order the deivce node with
                    nodes_with_para, new_ordered_node_list = insert_node_reorder_port_pool(component_orders,
                                                                                           nodes_with_para,
                                                                                           new_ordered_node_list,
                                                                                           node_with_para, node)
        # get the connection nodes linked device nodes
        origin_list_of_edge = json_file[target]["list_of_edge"]
        edge_list = filter_valid_edges(origin_list_of_edge)
        # clusters: connection nodes(int) and the list of it connected nodes
        conneted_nodes_cluster, _ = cluster_edges_for_connection_nodes(edge_list)
        connection_nodes = {}
        sort_connection_nodes = []
        # k:connection nodes(int) v: list of it connected nodes
        for k, v in conneted_nodes_cluster.items():
            connected_nodes_of_k = conneted_nodes_cluster[k]
            connected_device_nodes_reordered = []
            for connected_node in connected_nodes_of_k:
                # ports change to device nodes
                connected_node = connected_node.replace("-right", "")
                connected_node = connected_node.replace("-left", "")
                # get connected_node corresponding node_para form
                connected_node_idx = new_ordered_node_list.index(connected_node)
                conneted_device_node_with_para = nodes_with_para[connected_node_idx]
                order = component_orders[conneted_device_node_with_para]
                flag = None
                for i in range(len(connected_device_nodes_reordered)):
                    if order < component_orders[connected_device_nodes_reordered[i]]:
                        flag = 1
                        break
                if flag == 1:
                    connected_device_nodes_reordered.insert(i, conneted_device_node_with_para)
                else:
                    connected_device_nodes_reordered.append(conneted_device_node_with_para)
            # print(connected_device_nodes_reordered)
            str_connection_node = ""
            for tmp in connected_device_nodes_reordered:
                str_connection_node += tmp + "#"
            # print(str_connection_node)

            sort_connection_nodes.append(str_connection_node)
            connection_nodes[k] = str_connection_node
        # we order the connection node with alphabet order and add it to the whole node list
        sort_connection_nodes = sorted(sort_connection_nodes, key=str.lower)
        nodes_with_para = nodes_with_para + sort_connection_nodes
        # reorder the origin int formed connection node
        reorder_int_connection_node = []
        for k, v in conneted_nodes_cluster.items():
            idx_tmp = sort_connection_nodes.index(connection_nodes[k])
            flag = None
            for x in range(len(reorder_int_connection_node)):
                if idx_tmp < sort_connection_nodes.index(connection_nodes[reorder_int_connection_node[x]]):
                    flag = 1
                    break
            if flag == 1:
                reorder_int_connection_node.insert(i, k)
            else:
                reorder_int_connection_node.append(k)
        # print(reorder_int_connection_node)
        new_ordered_node_list = new_ordered_node_list + reorder_int_connection_node
        # print(new_ordered_node_list)

        idx_new_ordered_nodes_mapping = {}
        for n in range(len(new_ordered_node_list)):
            idx_new_ordered_nodes_mapping[new_ordered_node_list[n]] = n
        idxs_edge_list = []
        for edge in edge_list:
            edge[0] = edge[0].replace("-right", "")
            edge[0] = edge[0].replace("-left", "")
            idxs_edge_list.append([idx_new_ordered_nodes_mapping[edge[0]], idx_new_ordered_nodes_mapping[edge[1]]])

        # 2-D list sorting
        # [1,3][2,1][1,0]--> [1,0][1,3][2,1]
        idxs_edge_list = sorted(idxs_edge_list, key=(lambda x: [x[0], x[1]]))
        # print(idxs_edge_list)
        find_flag = 0

        for tmp_net in result_target_nets:
            # if tmp_net[0] == str(nodes_with_para) and tmp_net[1] == str(idxs_edge_list):
            if tmp_net[0] == str(nodes_with_para):
                find_flag = 1
                break
        if find_flag == 0:
            result_target_nets.append([str(nodes_with_para), str(idxs_edge_list)])
            result_targets.append(target)
        else:
            continue
    return result_targets


def write_no_isom_hash_to_file(no_isom_hash, no_isom_file_name="no_isom_hash.txt"):
    if no_isom_file_name:
        fo_file = open(no_isom_file_name, "w")
    for k, v in no_isom_hash.items():
        fo_file.write(k + '#' + str(v) + '\n')
    fo_file.close()
    return 0


def eliminate_isomorphism_for_instances(topo_instances, consider_edge=False):
    """
    :param topo_instances: a list of topologies which may has isomorphism
    :return: the subset of topo_instances that eliminated isomorphism
    """

    non_isomorphism_instances = []
    final_hash = {}
    for topo in topo_instances:
        isom_str = instance_to_isom_str(topo)
        if consider_edge:
            add_new = maintain_reward_hash_with_edge(final_hash, [isom_str, " "])
        else:
            add_new = maintain_reward_hash_no_edge(final_hash, [isom_str, " "])

        if add_new:
            non_isomorphism_instances.append(topo)

    return non_isomorphism_instances


def read_hash_to_instance():
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState

    args_file_name = "../config.py"

    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 6)
    need_add_node_list = []
    reward_hash = read_reward_hash(False)
    final_hash = {}
    topos = []
    for topo, info in reward_hash.items():
        sim.current = TopoGenState(init=True)
        topo_graph = get_topology_from_hash(topo)
        print(topo, topo_graph)
        edge_list = []
        for port_0, port_1s in topo_graph.items():
            for port_1 in port_1s:
                if [port_1, port_0] in edge_list:
                    continue
                edge_list.append([port_0, port_1])
        print(edge_list)
        # parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
        #               "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
        #               "GND": [0, 0, 1, 0, 0, 0, 0, 0],
        #               "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
        #               "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
        #               "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
        #               "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
        #               "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
        #               "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
        #               "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
        #               "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
        #               "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0]}
        # sim.current.parameters = parameters
        init_nodes = [0, 3, 1]
        for add_node in init_nodes:
            action = TopoGenAction('node', add_node)
            sim.act(action)
        for edge in edge_list:
            action = TopoGenAction('edge', edge)
            sim.act(action, False)
        topos.append(sim.get_state())
    return topos


def eliminate_isomorphism_all_test_3_com():
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState

    isom_sets = {}
    args_file_name = "../config.py"
    indices = list(range(1356))
    # indices = [8, 468, 471, 575, 630, 832]
    # indices = [29, 97, 171, 347, 348, 350, 405, 413, 476, 752, 820, 1005, 1039, 1106]
    # indices = [31, 73, 111, 236, 248, 268, 271, 511, 546, 629, 680, 725, 731, 771, 1011]
    # indices = [34, 152, 181, 269, 275, 285, 322, 358, 403, 419, 519, 596, 685, 810, 834, 947, 974, 1007, 1064, 1086, 1096, 1139, 1142, 1156, 1179, 1197, 1245, 1269, 1294, 1339, 1345, 1346]
    # indices = [213, 422, 648, 762, 895, 1061, 1090]
    # indices = [256, 1018]
    # indices = [259, 387, 399, 931, 1058]
    # indices = [54, 56, 62, 87, 162, 173, 221, 279, 313, 341, 357, 362, 365, 390, 416, 443, 446, 479, 481, 488, 490, 509, 510, 558, 559, 584, 653, 687, 694, 861, 864, 909, 913, 919, 941, 953, 973, 1006, 1031, 1056, 1067, 1092, 1094, 1095, 1100, 1190, 1194, 1199, 1221, 1297, 1319, 1329, 1331, 1347]
    # indices = [64, 483, 538, 981, 1192, 1200, 1209, 1284]
    # indices = [0,223]
    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 6)
    need_add_node_list = []
    # reward_hash = read_reward_hash(False)
    topo_reward = read_reward_hash_list()
    print(len(topo_reward))
    final_hash = {}
    topos = []
    for idx in indices:
        # for topo, info in reward_hash.items():
        print(topo_reward[idx])
        topo = topo_reward[idx][0]
        info = topo_reward[idx][1]
        sim.current = TopoGenState(init=True)
        topo_graph = get_topology_from_hash(topo)
        print(topo, topo_graph)
        edge_list = []
        for port_0, port_1s in topo_graph.items():
            for port_1 in port_1s:
                if [port_1, port_0] in edge_list:
                    continue
                edge_list.append([port_0, port_1])
        print(edge_list)
        parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
                      "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
                      "GND": [0, 0, 1, 0, 0, 0, 0, 0],
                      "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
                      "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
                      "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0]}
        sim.current.parameters = parameters
        init_nodes = [0, 3, 1]
        for add_node in init_nodes:
            action = TopoGenAction('node', add_node)
            sim.act(action)
        for edge in edge_list:
            action = TopoGenAction('edge', edge)
            sim.act(action, False)
        isom_str_set = instance_to_isom_str(sim.get_state())
        topo_info = str(isom_str_set[0]) + "|" + str(isom_str_set[1])
        print(topo_info)
        isom_node_list = isom_str_set[0]
        isom_node_list_str = str(isom_node_list)
        # sleep(1)
        # sim.get_state().visualize(str(idx), "isom_figures")
        topos.append(sim.get_state())
        add_new = maintain_reward_hash_with_edge(final_hash, [isom_str_set, str(info)])
        # add_new = maintain_reward_hash_no_edge(final_hash, [isom_str_set, str(info)])
        # if isom_sets.__contains__(isom_node_list_str):
        #     isom_sets[isom_node_list_str].append(idx)
        # else:
        #     isom_sets[isom_node_list_str] = [idx]
        if isom_sets.__contains__(topo_info):
            isom_sets[topo_info].append(idx)
        else:
            isom_sets[topo_info] = [idx]
    write_no_isom_hash_to_file(final_hash)
    print(final_hash)
    print("after eliminate ison:", len(final_hash), "previously it is: ", len(indices))
    # final_instances = eliminate_isomorphism_for_instances(topos, True)
    final_instances = eliminate_isomorphism_for_instances(topos, True)
    print("len of no isomorphism instances :", len(final_instances))
    print(len(isom_sets))
    for k, v in isom_sets.items():
        print(k, v)
    # for k,v in isom_sets.items():
    #     min_info = [100, 100, 100]
    #     max_info = [-1, -1, -1]
    #     print("topology", k)
    #     print("idx of this topology",v)
    #     for idx in v:
    #         print(idx,"info",topo_reward[idx][1])
    #         reward = float(topo_reward[idx][1][0])
    #         effi = float(topo_reward[idx][1][1])
    #         vout = float(topo_reward[idx][1][2])
    #         if reward < min_info[0]:
    #             min_info[0] = reward
    #         if reward > max_info[0]:
    #             max_info[0] = reward
    #         if effi < min_info[1]:
    #             min_info[1] = effi
    #         if effi > max_info[1]:
    #             max_info[1] = effi
    #         if vout < min_info[2]:
    #             min_info[2] = vout
    #         if vout > max_info[2]:
    #             max_info[2] = vout
    #     print("max and min:",max_info, min_info)
    #     print("diff:", [max_info[0]-min_info[0], max_info[1]-min_info[1], max_info[2]-min_info[2]])
    #     if max_info[0]-min_info[0] > 0.1 or max_info[1]-min_info[1] > 0.1 or max_info[2]-min_info[2] > 0.1:
    #         print("warning: Too large diff")

    return 0


def eliminate_isomorphism_shun_test_3_com():
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState

    isom_sets = {}
    args_file_name = "../config.py"
    indices = read_path_eliminate_isom_data()
    # indices = [1,46]

    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 6)
    need_add_node_list = []
    # reward_hash = read_reward_hash(False)
    topo_reward = read_reward_hash_list()
    final_hash = {}
    topos = []
    for idx in indices:
        # for topo, info in reward_hash.items():
        print(topo_reward[idx])
        topo = topo_reward[idx][0]
        info = topo_reward[idx][1]
        sim.current = TopoGenState(init=True)
        topo_graph = get_topology_from_hash(topo)
        print(topo, topo_graph)
        edge_list = []
        for port_0, port_1s in topo_graph.items():
            for port_1 in port_1s:
                if [port_1, port_0] in edge_list:
                    continue
                edge_list.append([port_0, port_1])
        print(edge_list)
        parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
                      "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
                      "GND": [0, 0, 1, 0, 0, 0, 0, 0],
                      "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
                      "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
                      "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0]}
        sim.current.parameters = parameters
        init_nodes = [0, 3, 1]
        for add_node in init_nodes:
            action = TopoGenAction('node', add_node)
            sim.act(action)
        for edge in edge_list:
            action = TopoGenAction('edge', edge)
            sim.act(action, False)
        isom_str_set = instance_to_isom_str(sim.get_state())
        isom_node_list = isom_str_set[0]
        isom_node_list_str = str(isom_node_list)
        # sleep(1)
        # sim.get_state().visualize(str(idx), "isom_figures")
        topos.append(sim.get_state())
        add_new = maintain_reward_hash_no_edge(final_hash, [isom_str_set, str(info)])
        if isom_sets.__contains__(isom_node_list_str):
            isom_sets[isom_node_list_str].append(idx)
        else:
            isom_sets[isom_node_list_str] = [idx]
    write_no_isom_hash_to_file(final_hash)
    print(final_hash)
    print("after eliminate ison:", len(final_hash), "previously it is: ", len(indices))
    final_instances = eliminate_isomorphism_for_instances(topos)
    print("len of no isomorphism instances :", len(final_instances))
    for k, v in isom_sets.items():
        print(k, v)
        for idx in v:
            # print(topo_reward[idx][1])
            print(topo_reward[idx][0])

    return 0


def eliminate_isomorphism_test_3_com():
    """
    test the eliminate_isomorphism using 3 component reward hash table
    :return:
    """
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState

    args_file_name = "../config.py"

    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 6)
    need_add_node_list = []
    reward_hash = read_reward_hash(False)
    # reward_hash = read_reward_hash_list()
    final_hash = {}
    topos = []
    for topo, info in reward_hash.items():
        sim.current = TopoGenState(init=True)
        topo_graph = get_topology_from_hash(topo)
        print(topo, topo_graph)
        edge_list = []
        for port_0, port_1s in topo_graph.items():
            for port_1 in port_1s:
                if [port_1, port_0] in edge_list:
                    continue
                edge_list.append([port_0, port_1])
        print(edge_list)
        parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
                      "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
                      "GND": [0, 0, 1, 0, 0, 0, 0, 0],
                      "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
                      "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
                      "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                      "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                      "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0]}
        sim.current.parameters = parameters
        init_nodes = [0, 3, 1]
        for add_node in init_nodes:
            action = TopoGenAction('node', add_node)
            sim.act(action)
        for edge in edge_list:
            action = TopoGenAction('edge', edge)
            sim.act(action, False)
        isom_str = instance_to_isom_str(sim.get_state())
        topos.append(sim.get_state())
        add_new = maintain_reward_hash_no_edge(final_hash, [isom_str, str(info)])
    write_no_isom_hash_to_file(final_hash)
    print(final_hash)
    print("after eliminate ison:", len(final_hash), "previously it is: ", len(reward_hash))
    final_instances = eliminate_isomorphism_for_instances(topos)
    print("len of no isomorphism instances :", len(final_instances))

    return 0


def eliminate_isomorphism_test_4_com(target_in_data):
    """
    test the eliminate isomorphism using 4 components
    :param target_in_data: list of PCC-xxxxxx to read in data_2.json
    :return:
    """
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState
    _, target_instances = get_target_instances(target_in_data, None, None)

    args_file_name = "../config.py"

    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 7)
    need_add_node_list = []
    need_add_edge_list = []
    init_nodes = [3, 0, 2, 2]
    for e in init_nodes:
        action = TopoGenAction('node', e)
        sim.act(action)
    edges = []
    edges = [[0, 3], [1, 9], [9, 7], [7, 5], [2, 6], [6, 10], [8, 4]]

    for edge in edges:
        action = TopoGenAction('edge', edge)
        sim.act(action)
    sim.current.parameters = copy.deepcopy(target_instances[0].parameters)
    topologies = [sim.get_state()]
    nets_to_ngspice_files(topologies, configs, configs['num_component'])
    simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
    effi = analysis_topologies(configs, len(topologies), configs['num_component'])
    print("effi-2", effi)
    print(instance_to_isom_str(sim.current))
    print(instance_to_isom_str(target_instances[0]))

    return 0


def get_target_instances(target_in_data, target_with_effi=None, del_isom=True):
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState
    file_name = "12-15-tmpresult.txt"
    fo_conf = open(file_name, "w")
    target_topologies = []
    target_topologies_2 = []
    target_topologies_3 = []
    mkdir("figures")
    mkdir("Results")

    args_file_name = "../config.py"

    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 7)
    json_file = json.load(open("jsons/data_2_tmp.json"))
    json_file = json.load(open("jsons/data_2.json"))
    tmp_json_file = copy.deepcopy(json_file)
    # print(target_in_data)
    fo_conf.write(str(target_in_data) + "\n")
    fo_conf.write("targets length " + str(len(target_in_data)) + "\n")
    filted_targets = target_in_data
    if del_isom:
        filted_targets = eliminate_isomorphism_for_json(target_in_data, tmp_json_file)
    print(len(filted_targets))
    fo_conf.write(str(filted_targets) + "\n")
    fo_conf.write("filted targets length " + str(len(filted_targets)) + "\n")

    # print(filted_targets)
    # print(len(filted_targets))
    # print(len(target_in_data))

    target_instances = []
    filter_vout_targets = []
    filter_vout_targets_2 = []
    filter_vout_targets_3 = []
    filted_targets_2 = []
    for target in filted_targets:
        file_name = target
        list_of_edge = json_file[target]["list_of_edge"]
        list_of_node = json_file[target]["list_of_node"]
        parameter_list = json_file[target]["parameter_list"]
        count_map = get_counts(json_file[target]["component_pool"])
        # print(json_file[target]["joint_list"])
        edge_list = filter_valid_edges(list_of_edge)
        need_add_edge_list = get_need_add_edges(edge_list, json_file[target]["port_2_idx"])

        sim.current = TopoGenState(init=True)
        sim.current.count_map = count_map
        sim.current.component_pool = json_file[target]["component_pool"]
        sim.current.port_2_idx = json_file[target]["port_2_idx"]
        sim.current.idx_2_port = change_key_2_int(json_file[target]["idx_2_port"])
        sim.current.port_pool = json_file[target]["port_pool"]
        sim.current.same_device_mapping = change_key_2_int(json_file[target]["same_device_mapping"])
        sim.current.comp2port_mapping = change_key_2_int(json_file[target]["comp2port_mapping"])
        sim.current.port2comp_mapping = change_key_2_int(json_file[target]["port2comp_mapping"])
        sim.current.parameters = json_file[target]["parameter_list"]
        # print(json_file[target]["netlist"])
        sim.current.component_pool = json_file[target]["component_pool"]
        # print(sim.current.parameters)

        sim.finish_node_set()
        for action in need_add_edge_list:
            sim.act(action, False)
            # sim.act(action)

        # topologies = [sim.get_state()]
        # nets_to_ngspice_files(topologies, configs, configs['num_component'])
        # simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        # effi = analysis_topologies(configs, len(topologies), configs['num_component'])
        # # print(target)
        # print("effi of topo:", effi[0])
        # if abs(target_with_effi[target]-effi[0]['efficiency']) > 0.01:
        #     print("seems wrong in simulate")
        # if effi[0].__contains__('output_voltage'):
        #     if effi[0]['output_voltage'] < 43 and effi[0]['efficiency'] > 0.8:
        #         filter_vout_targets.append(target)
        #         target_topologies.append(sim.get_state())
        #         fo_conf.write(str(target)+"\n")
        #     if effi[0]['output_voltage'] < 43 and effi[0]['efficiency'] > 0.6:
        #         filter_vout_targets_2.append(target)
        #     if effi[0]['output_voltage'] < 44 and effi[0]['efficiency'] > 0.6:
        #         filter_vout_targets_3.append(target)
        # else:
        #     continue
        target_topologies.append(sim.get_state())
        # if sim.current.component_pool.__contains__('capacitor-0'):
        #     target_topologies.append(sim.get_state())
        #     filted_targets_2.append(target)

    # print(filter_vout_targets)
    # print(filter_vout_targets_2)
    # print(filter_vout_targets_3)
    # print(target_topologies)
    # fo_conf.write(str(filter_vout_targets)+"\n")
    # fo_conf.close()
    # print(len(target_topologies))
    # print(len(filted_targets_2))
    return filted_targets, target_topologies
    # return filted_targets_2, target_topologies


def read_path_eliminate_isom_data():
    json_file = json.load(open("jsons/hash_merged.json"))
    cand_isom_topo_line_set = []
    for isom_set in json_file:
        indices = isom_set['indices']
        cand_isom_topo_line_set.append(int(indices[0]))
    print(len(cand_isom_topo_line_set), cand_isom_topo_line_set)

    print("")
    return cand_isom_topo_line_set


def try_cap_para(target_ins, candidate_cap_parameters, configs):
    """
    :param target_ins: the instance(just one) of the targets which has high efficiencies
    :param candidate_cap_parameters: candidate capacitor 0 parameters(list)
    :param configs: config.py file
    :return: the capacitor parameter with the highest reward
    """
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState
    para_efficiency_mapping = {}
    candidate_para_for_components = []
    para_tmp = copy.deepcopy(target_ins.parameters)
    t = 0
    highest_cap_para = 0
    highest_effi = []
    highest_reward = -1
    if not target_ins.parameters.__contains__("capacitor-0"):
        topologies = [target_ins]
        nets_to_ngspice_files(topologies, configs, configs['num_component'])
        simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        effi = analysis_topologies(configs, len(topologies), configs['num_component'])
        return target_ins, effi[0]
    for cap_para in candidate_cap_parameters:
        target_ins.parameters["capacitor-0"] = [0, 0, 0, cap_para, 0, 0, 0, 0]
        print(target_ins.parameters)
        topologies = [target_ins]
        nets_to_ngspice_files(topologies, configs, configs['num_component'])
        simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        effi = analysis_topologies(configs, len(topologies), configs['num_component'])
        print("effi of topo:", effi[0])
        if not effi[0].__contains__('output_voltage'):
            reward = 0
        else:
            reward = calculate_reward(effi[0], configs['target_vout'])
        if highest_reward < reward:
            highest_reward = reward
            highest_cap_para = cap_para
            highest_effi = effi[0]
    return highest_cap_para, highest_effi


def try_ind_para(target_ins, candidate_ind_parameters, configs):
    """
    :param target_ins: the instance(just one) of the targets which has high efficiencies
    :param candidate_ind_parameters: candidate inductor 0 parameters(list)
    :param configs: config.py file
    :return: the inductor parameter with the highest reward
    """
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState
    para_efficiency_mapping = {}
    candidate_para_for_components = []
    para_tmp = copy.deepcopy(target_ins.parameters)
    t = 0
    highest_ind_para = 0
    highest_effi = []
    highest_reward = -1
    if not target_ins.parameters.__contains__("inductor-0"):
        topologies = [target_ins]
        nets_to_ngspice_files(topologies, configs, configs['num_component'])
        simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        effi = analysis_topologies(configs, len(topologies), configs['num_component'])
        return target_ins, effi[0]
    for ind_para in candidate_ind_parameters:
        target_ins.parameters["inductor-0"] = [0, 0, 0, 0, ind_para, 0, 0, 0]
        print(target_ins.parameters)
        topologies = [target_ins]
        nets_to_ngspice_files(topologies, configs, configs['num_component'])
        simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        effi = analysis_topologies(configs, len(topologies), configs['num_component'])
        print("effi of topo:", effi[0])
        if not effi[0].__contains__('output_voltage'):
            reward = 0
        else:
            reward = calculate_reward(effi[0], configs['target_vout'])
        if highest_reward < reward:
            highest_reward = reward
            highest_ind_para = ind_para
            highest_effi = effi[0]

    return highest_ind_para, highest_effi


def adjusted_para_efficiency(target_ins, candidate_parameters, configs):
    """
    :param target_ins: the instance(just one) of the targets which has high efficiencies
    :param candidate_parameters: the dict with key: device, value: list of parameters that device can take
    :param configs: config.py file
    :return: dict with key:list of parameter set, value : the efficiency that instance take this parameter set
    """
    para_efficiency_mapping = {}
    candidate_para_for_components = []
    for component in target_ins.component_pool:
        if component not in ["GND", "VIN", "VOUT"]:
            para_of_k = sum(target_ins.parameters[component])
            component_type = component[:component.rfind('-')]
            if component_type == "FET-A":
                component_type = "FETA"
            if component_type == "FET-B":
                component_type = "FETB"
            candidate_para_of_comp = candidate_parameters[component_type]
            candidate_para_for_components.append(candidate_para_of_comp)
    print(candidate_para_for_components)
    para_tmp = copy.deepcopy(target_ins.parameters)
    t = 0
    for cand_para_set in product(*candidate_para_for_components):
        print(cand_para_set)
        # t += 1
        # if t > 3:
        #     break
        for i in range(len(cand_para_set)):
            current_para = target_ins.parameters[target_ins.component_pool[i + 3]]
            for j in range(len(current_para)):
                if current_para[j] > 0:
                    current_para[j] = cand_para_set[i]
                    break
            target_ins.parameters[target_ins.component_pool[i + 3]] = current_para
        print(target_ins.parameters)
        topologies = [target_ins]
        nets_to_ngspice_files(topologies, configs, configs['num_component'])
        simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        effi = analysis_topologies(configs, len(topologies), configs['num_component'])
        print("effi of topo:", effi[0])
        para_efficiency_mapping[str(cand_para_set)] = effi[0]
        target_ins.parameters = para_tmp
    return para_efficiency_mapping


def main(name):
    from ucts.TopoPlanner import TopoGenAction, TopoGenSimulator, TopoGenState

    isom_sets = {}
    args_file_name = "../config.py"
    sim_configs = get_sim_configs(configs)
    sim = TopoGenSimulator(sim_configs, 9)
    init_nodes = [0, 1, 3, 3, 3, 3]
    for add_node in init_nodes:
        action = TopoGenAction('node', add_node)
        sim.act(action)
    topos = []
    # edge_list = [[0, 5], [1, 3], [2, 7], [4, 6], [4, 8], [6, 8]]
    # edge_list = [[0, 7], [0, 9], [1, 3], [1, 5], [4, 8], [6, 10]]
    edge_list = [[0, 3], [0, 5], [4, 7], [6, 9], [8, 11], [10, 13], [1, 12], [1, 14]]
    for edge in edge_list:
        action = TopoGenAction('edge', edge)
        sim.act(action, False)
    parameters = {"VIN": [1, 0, 0, 0, 0, 0, 0, 0],
                  "VOUT": [0, 1, 0, 0, 0, 0, 0, 0],
                  "GND": [0, 0, 1, 0, 0, 0, 0, 0],
                  "FET-A-0": [0, 0, 0, 0, 0, 0.1, 0, 0],
                  "FET-B-0": [0, 0, 0, 0, 0, 0, 0.25, 0],
                  "FET-A-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                  "FET-A-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                  "FET-B-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                  "FET-B-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-0-left": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-0-right": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-0": [0, 0, 0, 0, 0.75, 0, 0, 0],
                  "inductor-1-left": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-1-right": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-1": [0, 0, 0, 0, 0.75, 0, 0, 0],
                  "inductor-2-left": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-2-right": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-2": [0, 0, 0, 0, 0.75, 0, 0, 0],
                  "inductor-3-left": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-3-right": [0, 0, 0, 0, 0, 0, 0, 1],
                  "inductor-3": [0, 0, 0, 0, 0.75, 0, 0, 0]}
    sim.current.parameters = parameters
    topos.append(sim.get_state())
    sim.get_state().visualize("test", "isom_figures")
    isom_str = instance_to_isom_str(sim.get_state())
    print(isom_str[0])
    print("")
    for n in range(len(isom_str[0])):
        print(isom_str[0][n], n, end=" | ")
    print("")
    print(isom_str[1])
    return

    # eliminate_isomorphism_test_4_com(target_in_data)
    # read_path_eliminate_isom_data()
    # return 0
    eliminate_isomorphism_all_test_3_com()
    # eliminate_isomorphism_shun_test_3_com()
    # eliminate_isomorphism_test_3_com()
    return 0
    # file_name = "result_final_targets.txt"
    # fo_conf = open(file_name, "w")
    # candidate_parameters = init_candidate_parameters()
    # history_file = "history.log"
    # target_in_data, target_with_effi = count_large_effi(history_file)
    # print(target_in_data)
    # print("targets has effi>0.1",len(target_in_data))
    # # target_in_data = ['PCC-000583', 'PCC-000919', 'PCC-002025']
    # # target_in_data = ['PCC-000583', 'PCC-000919', 'PCC-001055', 'PCC-001920', 'PCC-002025', 'PCC-002887', 'PCC-003460', 'PCC-003822']
    # filted_targets, target_instances = get_target_instances(target_in_data, target_with_effi, True)
    # print("after eliminate isom:", len(filted_targets))
    # return 0
    #
    #
    # configs = {}
    # args_file_name = "../config.py"
    # get_args(args_file_name, configs)
    # sim_configs = get_sim_configs(configs)
    # final_targets = []
    # final_instances = []
    # for i in range(len(target_instances)):
    #     instance = target_instances[i]
    #     # highest_ind_para, highest_effi = try_ind_para(instance, candidate_parameters['inductor'], configs)
    #     highest_ind_para, highest_effi = try_ind_para(instance, candidate_parameters['inductor'], configs)
    #     print(highest_ind_para, highest_effi)
    #     print(instance_to_isom_str(instance))
    #     if 20 < highest_effi['output_voltage'] < 30:
    #         fo_conf.write(str(filted_targets[i]) + '\n' + str(highest_ind_para) + '\n' + str(highest_effi) + '\n')
    #         isom_str, _ = instance_to_isom_str(instance)
    #         fo_conf.write(str(isom_str) + '\n')
    #         final_targets.append(filted_targets[i])
    #         final_instances.append(instance)
    #     print("-----------------------------------------------------------------------------")
    #
    # print(final_targets)
    # fo_conf.write(str(final_targets))
    # print(final_instances)
    #
    # for i in range(len(target_instances)):
    #     instance = target_instances[i]
    #     # highest_ind_para, highest_effi = try_ind_para(instance, candidate_parameters['inductor'], configs)
    #     highest_cap_para, highest_effi = try_cap_para(instance, candidate_parameters['capacitor'], configs)
    #     print(highest_cap_para, highest_effi)
    #     print(instance_to_isom_str(instance))
    #     if 20 < highest_effi['output_voltage'] < 30:
    #         fo_conf.write(str(filted_targets[i]) + '\n' + str(highest_cap_para) + '\n' + str(highest_effi) + '\n')
    #         isom_str, _ = instance_to_isom_str(instance)
    #         fo_conf.write(str(isom_str) + '\n')
    #         final_targets.append(filted_targets[i])
    #         final_instances.append(instance)
    #     print("-----------------------------------------------------------------------------")
    # print(final_targets)
    # fo_conf.write(str(final_targets))
    # print(final_instances)
    # fo_conf.close()
    # return
    # # for i in range(len(target_instances)):
    # #     target_ins = target_instances[i]
    # #     fo_conf.write(target_in_data[i] + "\n")
    # #     fo_conf.write(str(target_ins.component_pool) + "\n")
    # #     fo_conf.write(str(target_ins.graph) + "\n")
    # #     fo_conf.write(str(target_ins.port_pool) + "\n")
    # #     fo_conf.write(str(target_ins.parameters) + "\n")
    # #     para_efficiency_mapping = adjusted_para_efficiency(target_ins, candidate_parameters, configs)
    # #     for k, v in para_efficiency_mapping.items():
    # #         fo_conf.write(str(k) + " " + str(v) + "\n")
    # #     print("-----------------------------------------------------\n")
    # # print("exit")
    # # fo_conf.close()
    # # for topology in target_instances:
    # #     sorted_target_instances = sort_targets(target_instances)
    # #
    # # eliminate_isomorphism_for_json(target_in_data, json_file)
    #
    # exit(0)


# target_in_data = ['PCC-000583', 'PCC-000919', 'PCC-002025']#(0.9,45)
# target_in_data = ['PCC-000009', 'PCC-000117', 'PCC-000171', 'PCC-000215', 'PCC-000238', 'PCC-000248', 'PCC-000254', 'PCC-000279', 'PCC-000293', 'PCC-000319', 'PCC-000367', 'PCC-000522', 'PCC-000553', 'PCC-000583', 'PCC-000598', 'PCC-000608', 'PCC-000703', 'PCC-000720', 'PCC-000760', 'PCC-000819', 'PCC-000830', 'PCC-000846', 'PCC-000919', 'PCC-000924', 'PCC-000944', 'PCC-000998', 'PCC-001038', 'PCC-001055', 'PCC-001092', 'PCC-001172', 'PCC-001284', 'PCC-001334', 'PCC-001421', 'PCC-001450', 'PCC-001451', 'PCC-001550', 'PCC-001570', 'PCC-001571', 'PCC-001573', 'PCC-001661', 'PCC-001663', 'PCC-001768', 'PCC-001860', 'PCC-001879', 'PCC-001920', 'PCC-001927', 'PCC-001971', 'PCC-001973', 'PCC-001985', 'PCC-001986', 'PCC-001997', 'PCC-002025', 'PCC-002110', 'PCC-002121', 'PCC-002133', 'PCC-002153', 'PCC-002187', 'PCC-002192', 'PCC-002199', 'PCC-002208', 'PCC-002301', 'PCC-002352', 'PCC-002410', 'PCC-002425', 'PCC-002479', 'PCC-002498', 'PCC-002500', 'PCC-002564', 'PCC-002565', 'PCC-002640', 'PCC-002650', 'PCC-002679', 'PCC-002744', 'PCC-002776', 'PCC-002785', 'PCC-002799', 'PCC-002805', 'PCC-002826', 'PCC-002827', 'PCC-002887', 'PCC-002963', 'PCC-003041', 'PCC-003109', 'PCC-003126', 'PCC-003139', 'PCC-003184', 'PCC-003310', 'PCC-003365', 'PCC-003426', 'PCC-003460', 'PCC-003582', 'PCC-003606', 'PCC-003613', 'PCC-003614', 'PCC-003623', 'PCC-003634', 'PCC-003654', 'PCC-003778', 'PCC-003810', 'PCC-003822', 'PCC-003856', 'PCC-003876', 'PCC-004039', 'PCC-004047', 'PCC-004162', 'PCC-004285', 'PCC-004362', 'PCC-004364', 'PCC-004399', 'PCC-004439', 'PCC-004467', 'PCC-004512', 'PCC-004546', 'PCC-004558', 'PCC-004566', 'PCC-004570', 'PCC-004607', 'PCC-004618', 'PCC-004628', 'PCC-004648', 'PCC-004673', 'PCC-004747', 'PCC-004754', 'PCC-004784', 'PCC-004788', 'PCC-004825', 'PCC-004829', 'PCC-004860', 'PCC-004866', 'PCC-004868', 'PCC-004886', 'PCC-004911', 'PCC-004942', 'PCC-004967']
# target_in_data = ['PCC-000009', 'PCC-000171', 'PCC-000215', 'PCC-000248', 'PCC-000279', 'PCC-000293', 'PCC-000319', 'PCC-000367', 'PCC-000522', 'PCC-000553', 'PCC-000583', 'PCC-000598', 'PCC-000608', 'PCC-000703', 'PCC-000720', 'PCC-000760', 'PCC-000819', 'PCC-000830', 'PCC-000846', 'PCC-000919', 'PCC-000944', 'PCC-000998', 'PCC-001038', 'PCC-001055', 'PCC-001092', 'PCC-001172', 'PCC-001284', 'PCC-001334', 'PCC-001421', 'PCC-001450', 'PCC-001451', 'PCC-001550', 'PCC-001570', 'PCC-001571', 'PCC-001573', 'PCC-001661', 'PCC-001663', 'PCC-001768', 'PCC-001860', 'PCC-001879', 'PCC-001920', 'PCC-001927', 'PCC-001971', 'PCC-001973', 'PCC-001986', 'PCC-002025', 'PCC-002110', 'PCC-002121', 'PCC-002133', 'PCC-002153', 'PCC-002187', 'PCC-002199', 'PCC-002208', 'PCC-002301', 'PCC-002352', 'PCC-002410', 'PCC-002425', 'PCC-002479', 'PCC-002498', 'PCC-002500', 'PCC-002564', 'PCC-002565', 'PCC-002650', 'PCC-002679', 'PCC-002744', 'PCC-002776', 'PCC-002799', 'PCC-002805', 'PCC-002826', 'PCC-002827', 'PCC-002887', 'PCC-002963', 'PCC-003041', 'PCC-003126', 'PCC-003139', 'PCC-003184', 'PCC-003310', 'PCC-003365', 'PCC-003426', 'PCC-003460', 'PCC-003606', 'PCC-003613', 'PCC-003614', 'PCC-003623', 'PCC-003634', 'PCC-003654', 'PCC-003810', 'PCC-003822', 'PCC-003856', 'PCC-003876', 'PCC-004047', 'PCC-004162', 'PCC-004285', 'PCC-004362', 'PCC-004364', 'PCC-004399', 'PCC-004439', 'PCC-004467', 'PCC-004512', 'PCC-004546', 'PCC-004558', 'PCC-004566', 'PCC-004570', 'PCC-004607', 'PCC-004618', 'PCC-004628', 'PCC-004648', 'PCC-004673', 'PCC-004747', 'PCC-004754', 'PCC-004784', 'PCC-004788', 'PCC-004825', 'PCC-004829', 'PCC-004860', 'PCC-004866', 'PCC-004868', 'PCC-004886', 'PCC-004911', 'PCC-004942', 'PCC-004967']
# target_in_data = ['PCC-000293', 'PCC-000319', 'PCC-000760', 'PCC-001055', 'PCC-001092', 'PCC-001550', 'PCC-001571', 'PCC-001573', 'PCC-001879', 'PCC-001920', 'PCC-001927', 'PCC-002025', 'PCC-002110', 'PCC-002744', 'PCC-002805', 'PCC-002827', 'PCC-002887', 'PCC-002963', 'PCC-003126', 'PCC-003460', 'PCC-003614', 'PCC-003822', 'PCC-004285', 'PCC-004570', 'PCC-004628', 'PCC-004866', 'PCC-004911']
# target_in_data = ['PCC-000583', 'PCC-000919', 'PCC-001055', 'PCC-001920', 'PCC-002025', 'PCC-002887', 'PCC-003460', 'PCC-003822']
if __name__ == '__main__':
    main('PyCharm')
