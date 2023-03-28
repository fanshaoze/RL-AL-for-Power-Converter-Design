import collections
import copy
import csv
import os
import shutil
import random
import hashlib
import math
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


def del_all_files(root_dir):
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exits = os.path.exists(path)
    if not is_exits:
        os.makedirs(path)
        print(path + ' created')
        return True
    else:
        print(path + ' already existed')
        return False


def get_sim_configs(configs):
    sim_configs = deepcopy(configs)
    return sim_configs


def init_position(position_number):
    init_position_X = []
    init_position_Y = []
    initFood = []
    init_nums = position_number

    init_size = 0
    ran_size = 0
    while True:
        init_flag = 0
        tmp_x = random.randint(0, 4)
        tmp_y = random.randint(0, 4)
        tmp_food = random.randint(0, 4)
        ran_size += 1
        for init_index in range(0, len(init_position_X)):
            if (init_position_X[init_index] == tmp_x and init_position_Y[init_index] == tmp_y) \
                    or tmp_y == tmp_food:
                init_flag = 1
                break
        if init_flag == 1:
            continue
        else:
            init_position_X.append(tmp_x)
            init_position_Y.append(tmp_y)
            initFood.append(tmp_food)
            init_size += 1
            if init_size == init_nums:
                break
    # (self, _sim, _maxDepth, _numRuns, _ucbScalar, _gamma, _leafValue, _endEpisodeValue):
    position_list = []
    for i in range(0, init_size):
        position_list.append((init_position_X[i], init_position_Y[i], initFood[i]))
    return position_list


def generate_depth_list(start, end, step_len):
    max_depth = []
    for i in range(start, end, step_len):
        max_depth.append(i)
    return max_depth


def generate_traj_List(start, end, step_len):
    traj = []
    for i in range(start, end, step_len):
        traj.append(i)
    return traj


def save_results_to_csv(out_file_name, result_rows):
    with open(out_file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(result_rows)
    f.close()


def save_reward_hash(sim):
    hash_file_name = "reward_hash.txt"
    hash_fo = open(hash_file_name, "w")
    for (k, v) in sim.graph_2_reward.items():
        hash_fo.write(str(k) + '#' + str(v) + '\n')
    hash_fo.close()


def save_one_in_hash_file(topology, efficiency, hash_file_name="reward_hash.txt"):
    hash_fo = open(hash_file_name, "a")
    hash_fo.write(str(topology) + '$' + str(efficiency) + '\n')
    hash_fo.close()


def hash_str_encoding(graph_str):
    m = hashlib.md5()
    m.update(graph_str.encode('utf-8'))
    return m.hexdigest()


def read_reward_hash_list(hash_file_name="reward_hash.txt"):
    graph_2_reward_tmp = []
    fo_conf = open(hash_file_name, "r")
    # line = fo_conf.readline()
    while True:
        line = fo_conf.readline()
        # print(line)
        if not line:
            break
        key_value = line.split('$')
        topo = key_value[0]

        str_list = key_value[1][1:-2]
        # print(str_list)
        value_str_list = str_list.split(',')
        hash_values = []
        for every_value in value_str_list:
            # print(every_value)
            hash_values.append(float(every_value))
        graph_2_reward_tmp.append([topo, hash_values])
    fo_conf.close()
    return graph_2_reward_tmp


def read_reward_hash(hash_file_name="reward_hash.txt", md5_trans=False):
    graph_2_reward_tmp = {}
    # hash_file_name = "reward_hash.txt"
    fo_conf = open(hash_file_name, "r")
    line = fo_conf.readline()
    while True:
        line = fo_conf.readline()
        # print(line)
        if not line:
            break
        key_value = line.split('$')
        topo = key_value[0]
        if md5_trans:
            md5_topo = hash_str_encoding(topo)
        str_list = key_value[1][1:-2]
        # print(str_list)
        value_str_list = str_list.split(',')
        hash_values = []
        for every_value in value_str_list:
            # print(every_value)
            hash_values.append(float(every_value))
        if md5_trans:
            graph_2_reward_tmp[md5_topo] = hash_values
        else:
            graph_2_reward_tmp[topo] = hash_values
    fo_conf.close()
    return graph_2_reward_tmp


def get_topology_from_hash(topo_str):
    conns = topo_str.split("],")
    # print(conns)
    graph = collections.defaultdict(set)
    for conn in conns:
        conn = conn.split(":")
        # print(conn[0])
        try:
            start_port = int(conn[0])
            end_port_list = conn[1][1:]
            # print(end_port_list)
            end_ports = end_port_list.split(',')
            for end_port in end_ports:
                pass
                graph[start_port].add(int(end_port))
                graph[int(end_port)].add(start_port)
        except:
            continue
    return graph


def get_steps_traj(total_traj, total_step, fix_traj, multi, fix_step_traj=False):
    steps_traj = []
    if fix_step_traj:
        traj = total_traj / total_step
        for i in range(total_step):
            steps_traj.append(traj)
        print(steps_traj)
        return steps_traj
    fix_step_number = 2
    for i in range(fix_step_number):
        steps_traj.insert(0, fix_traj)
    decrease_traj = total_traj - fix_step_number * fix_traj
    last_traj = decrease_traj * (multi - 1) / (math.pow(multi, total_step - fix_step_number) - 1)

    traj = last_traj
    for i in range(total_step - fix_step_number):
        if traj < fix_traj:
            steps_traj.insert(0, fix_traj)
        else:
            steps_traj.insert(0, int(traj) + 1)
        traj *= multi
    for i in range(len(steps_traj)):
        if steps_traj[i] == 0:
            steps_traj[i] = 1
    steps_traj[0] += total_traj - sum(steps_traj)

    return steps_traj


def Merge(dict1, dict2):
    dict2.update(dict1)
    return dict2


#
# def read_approve_path(file_name='approve_path_vout.txt'):
#     approve_path = []
#     path_fo = open(file_name, "r")
#     while True:
#         line = path_fo.readline()
#         if not line:
#             break
#         lines = line.split(':')
#         path_str = lines[1]
#         line = path_fo.readline()
#         lines = line.split(':')
#         freq_str = lines[1]
#         approve_path.append((path_str[1:-1], float(freq_str[1:-1])))
#     print("path with frequency:")
#     for path_freq in approve_path:
#         print(path_freq)
#     return approve_path


def read_approve_path(lower_bound=0.0, file_name='5comp_path_freqs.json'):
    with open(file_name) as f:
        data = json.load(f)

    high_paths = []
    for i in range(len(data)):
        if data[i][1] > lower_bound:
            high_paths.append((data[i][0], data[i][1]))
    # for ele in high_paths:
    #     print(ele)
    f.close()
    return high_paths


def sort_components(components):
    priorities = []
    # component_2_priority = {'Sa': 0, 'Sb': 1, 'L': 2, 'C': 3}
    component_2_priority = {'Sa': 0, 'Sb': 1, 'L': 2, 'C': 3}
    # priority_2_component = {0: 'Sa', 1: 'Sb', 2: 'L', 3: 'C'}
    priority_2_component = {0: 'Sa', 1: 'Sb', 2: 'L', 3: 'C'}
    for component in components:
        priorities.append(component_2_priority[component])
    priorities.sort()
    components = []
    for priority in priorities:
        components.append(priority_2_component[priority])
    return components


def read_joint_component_prob(num_component, basic_components=None,
                              file_name='5comp_node_joint_probs.json'):
    if basic_components is None:
        basic_components = ['Sa', 'Sb', 'L', 'C']
    component_condition_prob = []
    component_prob_dict = {}
    component_condition_prob = {}
    with open(file_name) as f:
        data = json.load(f)
    i = 0
    sub_graphs = [[]]
    for sub_graph in sub_graphs:

        for component in basic_components:
            tmp_sub_graph = deepcopy(sub_graph)
            tmp_sub_graph.append(component)
            if len(tmp_sub_graph) > num_component:
                break
            sub_graphs.append(tmp_sub_graph)
        if len(tmp_sub_graph) > num_component:
            break
    # print(sub_graphs)
    # for sub_graph_prob in data:
    #     component_prob_dict[str(sort_components(sub_graph_prob[0]))] = sub_graph_prob[1]
    # # for k, v in component_prob_dict.items():
    # #     print(k, ":", v)
    # component_prob_dict[str([])] = 1
    component_prob_dict = simplify_compoent_prob(basic_components, num_component, file_name)
    for sub_graph in sub_graphs:
        # print("")
        sum_freq = 0
        if len(sub_graph) < num_component:
            if str(sort_components(sub_graph)) not in component_prob_dict:
                for component in basic_components:
                    tmp_sub_graph = deepcopy(sub_graph)
                    tmp_sub_graph.append(component)
                    if len(tmp_sub_graph) <= num_component:
                        condition_str = str(sort_components(tmp_sub_graph)) + '|' + \
                                        str(sort_components(sub_graph))
                        component_condition_prob[condition_str] = 0.01
                        sum_freq += 0.01
            else:
                for component in basic_components:
                    tmp_sub_graph = deepcopy(sub_graph)
                    tmp_sub_graph.append(component)
                    if len(tmp_sub_graph) <= num_component:
                        condition_str = str(sort_components(tmp_sub_graph)) + '|' + \
                                        str(sort_components(sub_graph))
                        if str(sort_components(tmp_sub_graph)) in component_prob_dict:
                            condition_freq = component_prob_dict[str(sort_components(tmp_sub_graph))] \
                                             / component_prob_dict[str(sort_components(sub_graph))]
                        else:
                            condition_freq = 0.01
                        component_condition_prob[condition_str] = condition_freq
                        sum_freq += condition_freq
            for component in basic_components:
                tmp_sub_graph = deepcopy(sub_graph)
                tmp_sub_graph.append(component)
                if len(tmp_sub_graph) <= num_component:
                    condition_str = str(sort_components(tmp_sub_graph)) + '|' + \
                                    str(sort_components(sub_graph))
                    component_condition_prob[condition_str] = component_condition_prob[condition_str] / sum_freq
                    # print(condition_str, ':', component_condition_prob[condition_str])
    # for k, v in component_condition_prob.items():
    #     print(k, ":", v)
    return component_condition_prob


# def sample_component(sub_graph, joint_component_prob, allowed_actions, basic_components=None):
#     if basic_components is None:
#         basic_components = ["Sa", "Sb", "C", "L"]
#     tmp_sub_graph = deepcopy(sub_graph)
#     tmp_sub_graph.remove('VOUT')
#     tmp_sub_graph.remove('VIN')
#     tmp_sub_graph.remove('GND')
#     for i in range(len(tmp_sub_graph)):
#         if 'Sa' in tmp_sub_graph[i]:
#             tmp_sub_graph[i] = 'Sa'
#         elif 'Sb' in tmp_sub_graph[i]:
#             tmp_sub_graph[i] = 'Sb'
#         elif 'L' in tmp_sub_graph[i]:
#             tmp_sub_graph[i] = 'L'
#         elif 'C' in tmp_sub_graph[i]:
#             tmp_sub_graph[i] = 'C'
#     sample_prob = random.random()
#     sum_prob = 0
#     for action in allowed_actions:
#         component = basic_components[action.value]
#         append_sub_graph = deepcopy(tmp_sub_graph)
#         append_sub_graph.append(component)
#         condition_str = str(sort_components(append_sub_graph)) + '|' + str(sort_components(tmp_sub_graph))
#         if sum_prob < sample_prob <= sum_prob + joint_component_prob[condition_str]:
#             # print(condition_str)
#             return action
#         else:
#             sum_prob += joint_component_prob[condition_str]
#     return None

def sample_component(sub_graph, joint_component_prob, allowed_actions, basic_components=None):
    if basic_components is None:
        basic_components = ["Sa", "Sb", "L", "C"]
    tmp_sub_graph = deepcopy(sub_graph)
    tmp_sub_graph.remove('VOUT')
    tmp_sub_graph.remove('VIN')
    tmp_sub_graph.remove('GND')
    for i in range(len(tmp_sub_graph)):
        if 'Sa' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'Sa'
        elif 'Sb' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'Sb'
        elif 'L' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'L'
        elif 'C' in tmp_sub_graph[i]:
            tmp_sub_graph[i] = 'C'
    sample_prob = random.random()
    sum_prob = 0
    action_weights = []
    basic_weight = 0.1
    sum_weight = 0
    for action in allowed_actions:
        component = basic_components[action.value]
        append_sub_graph = deepcopy(tmp_sub_graph)
        append_sub_graph.append(component)
        condition_str = str(sort_components(append_sub_graph)) + '|' + str(sort_components(tmp_sub_graph))
        action_weights.append(joint_component_prob[condition_str])

    print(action_weights)
    # In case all the prob is 0, which means random
    if sum(action_weights) == 0:
        action_weights = [1 for _ in action_weights]
    if action_weights:
        action_list = random.choices(allowed_actions, weights=action_weights, k=1)
        return action_list[0]
    else:
        return None


def simplify_compoent_prob(basic_components, num_component, file_name='5comp_node_joint_probs.json'
                           ):
    if basic_components is None:
        basic_components = ['Sa', 'Sb', 'L', 'C']
    sub_graphs = [[]]
    component_counts = {'Sa': 0, 'Sb': 0, 'L': 0, 'C': 0}
    sub_graphs_with_count = [[]]
    for sub_graph in sub_graphs:
        for component in basic_components:
            tmp_sub_graph = deepcopy(sub_graph)
            tmp_sub_graph.append(component)

            if len(tmp_sub_graph) > num_component:
                break
            sub_graph_with_count = []
            count_Sa = tmp_sub_graph.count('Sa')
            for x in range(count_Sa):
                sub_graph_with_count.append('Sa' + str(x))
            count_Sb = tmp_sub_graph.count('Sb')
            for x in range(count_Sb):
                sub_graph_with_count.append('Sb' + str(x))
            count_L = tmp_sub_graph.count('L')
            for x in range(count_L):
                sub_graph_with_count.append('L' + str(x))
            count_C = tmp_sub_graph.count('C')
            for x in range(count_C):
                sub_graph_with_count.append('C' + str(x))
            if sub_graph_with_count in sub_graphs_with_count:
                continue
            else:
                sub_graphs_with_count.append(sub_graph_with_count)
                sub_graphs.append(tmp_sub_graph)
        if len(sub_graph) > num_component:
            break
    # for sub_graph in sub_graphs_with_count:
    #     print(sub_graph)
    simplified_data = {}
    with open(file_name) as f:
        data = json.load(f)
    for sub_graph_with_count in sub_graphs_with_count:
        if sub_graph_with_count == []:
            simplified_data[str(sub_graph_with_count)] = 1
            continue
        for sub_graph_prob in data:
            if set(sub_graph_prob[0]) == set(sub_graph_with_count):
                for n in range(len(sub_graph_with_count)):
                    if 'Sa' in sub_graph_with_count[n]:
                        sub_graph_with_count[n] = 'Sa'
                    elif 'Sb' in sub_graph_with_count[n]:
                        sub_graph_with_count[n] = 'Sb'
                    elif 'L' in sub_graph_with_count[n]:
                        sub_graph_with_count[n] = 'L'
                    elif 'C' in sub_graph_with_count[n]:
                        sub_graph_with_count[n] = 'C'
                simplified_data[str(sub_graph_with_count)] = sub_graph_prob[1]
                break
    # for simp_data in simplified_data:
    #     print(simp_data)
    return simplified_data


def get_component_type(item):
    if 'Sa' in item:
        return 'Sa'
    elif 'Sb' in item:
        return 'Sb'
    elif 'L' in item:
        return 'L'
    elif 'C' in item:
        return 'C'
    elif 'GND' in item:
        return 'GND'
    elif 'VIN' in item:
        return 'VIN'
    elif 'VOUT' in item:
        return 'VOUT'


def get_component_count_suffix(item):
    if 'Sa' in item:
        return int(item.replace('Sa', ''))
    elif 'Sb' in item:
        return int(item.replace('Sb', ''))
    elif 'L' in item:
        return int(item.replace('L', ''))
    elif 'C' in item:
        return int(item.replace('C', ''))
    elif 'GND' in item:
        return int(item.replace('GND', ''))
    elif 'VIN' in item:
        return int(item.replace('VIN', ''))
    elif 'VOUT' in item:
        return int(item.replace('VOUT', ''))


def plot_topo_counts(idx, count, save_name):
    # idx = [1, 2, 3, 4, 5]
    # count = [6, 7, 6, 1, 2]
    plt.bar(idx, count)
    plt.title('count')
    # plt.show()
    plt.savefig(save_name)
    plt.close()


def read_sta_result(file_name):
    data_json_file = json.load(open(file_name))
    return data_json_file


def save_sta_json(simu_results, file_name):
    with open(file_name, 'w') as f:
        json.dump(simu_results, f)
    f.close()


def save_sta_result(sta_list, count_file_name):
    count_fo = open(count_file_name, "w")
    for i in range(len(sta_list)):
        count_fo.write(str(i) + '\t' + str(sta_list[i][1]) + '\n')
    count_fo.close()


def save_count_sta_result(sta_list, count_file_name, group_count):
    count_fo = open(count_file_name, "w")
    i = 0
    sum_topo = 0
    group_count = group_count + 5000
    while i < 200000:
        if i > 0 and i % group_count == 0:
            count_fo.write(str(i) + '\t' + str(sum_topo / group_count) + '\n')
            sum_topo = 0
        if (len(sta_list) - i) >= group_count:
            for _ in range(group_count):
                sum_topo += sta_list[i][1]
                i += 1
    sum_topo = 0
    group_count = group_count - 9000
    while i < len(sta_list):
        if i > 0 and i % group_count == 0:
            count_fo.write(str(i) + '\t' + str(sum_topo / group_count) + '\n')
            sum_topo = 0
        if (len(sta_list) - i) >= group_count:
            for _ in range(group_count):
                sum_topo += sta_list[i][1]
                i += 1
        else:
            j = 0
            for _ in range(len(sta_list) - i):
                sum_topo += sta_list[i][1]
                i += 1
                j += 1
    count_fo.write(str(i) + '\t' + str(sum_topo / j) + '\n')

    count_fo.close()


def merge_sta_result(files, target_file):
    sta_all = {}
    for count_file_name in files:
        data_json_file = json.load(open(count_file_name))
        for k, v in data_json_file.items():
            if k in sta_all:
                sta_all[k][0] += v[0]
            else:
                sta_all[k] = v
    save_sta_json(sta_all, target_file)


def get_sta_reward_prob(sta_BF, sta_UCT, prob_sta_BF_file, prob_sta_UCT_file):
    UCT_probs = []
    BF_probs = []
    for i in range(20):
        count_UCT = 0
        count_BF = 0
        stan_reward = i / 20
        print('sta_UCT', len(sta_UCT))

        for k, v in sta_UCT.items():
            if stan_reward < v[1] <= stan_reward + 0.05:
                count_UCT += 1
        UCT_probs.append([stan_reward, count_UCT])
        for k, v in sta_BF.items():
            if stan_reward < v[1] <= stan_reward + 0.05:
                count_BF += 1
        BF_probs.append([stan_reward, count_BF])

    count_fo = open(prob_sta_UCT_file, "w")
    for i in range(len(UCT_probs)):
        count_fo.write(
            '(' + str(UCT_probs[i][0]) + '~' + str(UCT_probs[i][0] + 0.5) + ']' + '\t' + str(UCT_probs[i][1]) + '\n')
    count_fo.close()
    count_fo = None
    count_fo = open(prob_sta_BF_file, "w")
    for i in range(len(BF_probs)):
        count_fo.write(
            '(' + str(BF_probs[i][0]) + ',' + str(BF_probs[i][0] + 0.5) + ']' + '\t' + str(BF_probs[i][1]) + '\n')
    count_fo.close()


def remove_tmp_files():
    os.system("rm *-analytic*")
    os.system("rm PCC-*.cki")
    os.system("rm PCC-*.simu")
    os.system("rm PCC-*.simu")
    os.system("rm ./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/database/PCC*.json")


def read_stas(BF_file_name, sta_file_name):
    sta_BF = read_sta_result(BF_file_name)
    sta_UCT = read_sta_result(sta_file_name)
    tmp_sta_BF = copy.deepcopy(sta_BF)
    tmp_sta_UCT = copy.deepcopy(sta_UCT)

    for k, v in sta_BF.items():
        if k not in tmp_sta_UCT:
            tmp_sta_UCT[k] = [0, v[1], v[2]]
    for k, v in sta_UCT.items():
        if k not in tmp_sta_BF:
            tmp_sta_BF[k] = [0, v[1], v[2]]
    print('sta_UCT', len(tmp_sta_UCT))

    get_sta_reward_prob(tmp_sta_BF, tmp_sta_UCT, "sta_BF_prob.txt", "sta_UCT_prob.txt")

    BF_sta_list = []
    for k, v in tmp_sta_BF.items():
        BF_sta_list.append([k, v[0]])

    def keyFunc(element):
        return element[1]

    BF_sta_list.sort(key=keyFunc)
    UCT_sta_list = []
    for topo_count in BF_sta_list:
        # print(topo_count)
        count_info = tmp_sta_UCT[topo_count[0]]
        UCT_sta_list.append([topo_count[0], count_info[0]])

    sum_count = 0
    for sta in BF_sta_list:
        sum_count += sta[1]
    print('BF_count:', sum_count)
    sum_count = 0
    for sta in UCT_sta_list:
        sum_count += sta[1]
    print('only_epr_count:', sum_count)
    count_file_name = "BF_sta.txt"
    save_sta_result(BF_sta_list, count_file_name)
    count_file_name = "UCT_sta.txt"
    save_sta_result(UCT_sta_list, count_file_name)

    group_count = 5000
    count_file_name = "BF_sta_" + str(group_count) + ".txt"
    save_count_sta_result(BF_sta_list, count_file_name, group_count)
    count_file_name = "UCT_sta_" + str(group_count) + ".txt"
    save_count_sta_result(UCT_sta_list, count_file_name, group_count)
    # plot_topo_counts([i for i in range(len(BF_sta_list))], [j[1] for j in BF_sta_list], "BF_sta")
    # plot_topo_counts([i for i in range(len(UCT_sta_list))], [j[1] for j in UCT_sta_list], "UCT_sta")


get_steps_traj(9 * 80, 9, 6, 2, False)
# file_name = '5comp_node_joint_probs.json'
# with open(file_name) as f:
#     data = json.load(f)
# for i in data:
#     print(i)
# simplify_compoent_prob(None, 5)
# list_a = [1, 0, -1]
# list_b = [1, -1, 0]
# read_joint_component_prob(5, None)
# read_approve_path()
# print("..........................")
# read_approve_path()

# plot_topo_counts(None, None)
# read_stas('sta_BF_all.json', 'sta_only_epr_all.json')

# files_BF = ['sta_BF.json']
# files_UCT = ['sta_only_epr.json']
# for i in range(5):
#     files_BF.append('stas_BF/sta_BF_'+str(i)+'.json')
#     files_UCT.append('stas_UCT/sta_only_epr_'+str(i)+'.json')
# merge_sta_result(files_BF, 'sta_BF_all.json')
# merge_sta_result(files_UCT, 'sta_only_epr_all.json')
