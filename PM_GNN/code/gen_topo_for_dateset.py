import collections
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import random
import time
import json
import os
import ast
from collections import OrderedDict
import itertools as it
import subprocess
import csv
import numpy as np
import copy as cp
from ast import literal_eval
from PM_GNN.code.utils_new.graphUtils import indexed_graph_to_adjacency_matrix, adj_matrix_to_graph, graph_to_adjacency_matrix, \
    nodes_and_edges_to_adjacency_matrix, adj_matrix_to_edges


class TopoGraph(object):
    def __init__(self, node_list, adj_matrix=None, graph=None, edge_list=None, hint=None, params=None):
        self.node_list = node_list

        if adj_matrix is not None:
            self.adj_matrix = adj_matrix
        elif graph is not None:
            if hint == 'indexed':
                self.adj_matrix = indexed_graph_to_adjacency_matrix(graph)
            else:
                self.adj_matrix = graph_to_adjacency_matrix(graph, node_list)
        elif edge_list is not None:
            self.adj_matrix = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)
        else:
            raise Exception('failed to initialize Graph')

        if params is not None:
            self.params = params
        else:
            self.params = {}

    def modify_port(self, port: str) -> str:
        """
        Merge ports for topo_analysis
        """
        if port.startswith('inductor'):
            ret = 'inductor'
        elif port.startswith('capacitor'):
            ret = 'capacitor'
        elif port.startswith('FET-A'):
            ret = 'FET-A'
        elif port.startswith('FET-B'):
            ret = 'FET-B'
        else:
            ret = port

        if port in self.params:
            ret += ' ' + str(self.params[port])

        return ret

    def find_paths(self, source: int, target: int, exclude=[]) -> list:
        """
        Return a list of paths from source to target without reaching `exclude`.

        :param adj_matrix: the adjacency matrix of a graph
        :param exclude: nodes in this list are excluded from the paths (e.g. VIN to VOUT *without* reaching GND)
        """
        node_num = len(self.adj_matrix)

        paths = []

        def dfs(s, t, cur_path):
            """
            Perform dfs starting from s to find t, excluding nodes in exclude.
            cur_path stores the node visited on the current path.
            Results are added to paths.
            """
            if s in exclude:
                return

            if s == t:
                paths.append(cur_path + [s])
                return

            for neighbor in range(node_num):
                # find neighbors that are not visited in this path
                if neighbor != s and self.adj_matrix[s][neighbor] == 1 and not neighbor in cur_path:
                    dfs(neighbor, t, cur_path + [s])

        dfs(source, target, [])

        return paths

    def find_end_points_paths(self):
        """
        Find paths between any of VIN, VOUT, GND
        """
        gnd = self.node_list.index('GND')
        vin = self.node_list.index('VIN')
        vout = self.node_list.index('VOUT')

        paths = self.find_paths(vin, vout, [gnd]) + \
                self.find_paths(vin, gnd, [vout]) + \
                self.find_paths(vout, gnd, [vin])

        return paths

    def encode_path_as_string(self, path):
        """
        Convert a path to a string, so it's hashbale and readable
        """
        # 1. convert to node list
        path = [self.node_list[idx] for idx in path]

        # 2. drop connection nodes
        path = list(filter(lambda port: not isinstance(port, int), path))

        for index, item in enumerate(path):
            if item not in ['VIN', 'VOUT', 'GND']:
                path[index] = path[index][:-1]

        # 3. merge ports with different ids

        path = [self.modify_port(port) for port in path]

        # 4. to string
        path = ' - '.join(path)

        return path

    def find_end_points_paths_as_str(self):
        paths = self.find_end_points_paths()
        paths_str = [self.encode_path_as_string(path) for path in paths]

        return paths_str

    def eliminate_redundant_comps(self):
        """
        Remove redundant components in the adjacency matrix.
        """
        node_num = len(self.node_list)
        paths = self.find_end_points_paths()

        # compute traversed nodes
        traversed_nodes = set()
        for path in paths:
            traversed_nodes.update(path)
        traversed_nodes = list(traversed_nodes)

        new_matrix = \
            [[self.adj_matrix[i][j] for j in range(node_num) if j in traversed_nodes]
             for i in range(node_num) if i in traversed_nodes]
        new_node_list = [self.node_list[idx] for idx in traversed_nodes]

        self.adj_matrix = new_matrix
        self.node_list = new_node_list

    def get_graph(self):
        return adj_matrix_to_graph(self.node_list, self.adj_matrix)

    def get_edge_list(self):
        return adj_matrix_to_edges(self.node_list, self.adj_matrix)

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_node_list(self):
        return self.node_list


def union(x, y, parent):
    f_x = find(x, parent)
    f_y = find(y, parent)

    if f_x == f_y:
        return False

    parent[f_x] = f_y

    return True


def find(x, parent):
    if parent[x] != x:
        parent[x] = find(parent[x], parent)

    return parent[x]


def initial(n):
    # number of basic component in topology

    component_pool = ["GND", 'VIN', 'VOUT']

    port_pool = ["GND", 'VIN', 'VOUT']

    basic_compoments = ["Sa", "Sb", "C", "L"]

    #    basic_compoments = ["R"]

    count_map = {"Sa": 0, "Sb": 0, "C": 0, "L": 0}

    #    count_map = {"R":0}

    comp2port_mapping = {0: [0], 1: [1], 2: [2]}  # key is the idx in component pool, value is idx in port pool

    port2comp_mapping = {0: 0, 1: 1, 2: 2}

    index = range(len(basic_compoments))

    port_2_idx = {"GND": 0, 'VIN': 1, 'VOUT': 2}

    idx_2_port = {0: 'GND', 1: 'VIN', 2: "VOUT"}

    same_device_mapping = {}

    graph = collections.defaultdict(set)

    for i in range(n):
        idx = random.choice(index)

        count = str(count_map[basic_compoments[idx]])

        count_map[basic_compoments[idx]] += 1

        component = basic_compoments[idx] + count

        component_pool.append(component)

        idx_component_in_pool = len(component_pool) - 1

        port_pool.append(component + '-left')

        port_pool.append(component + '-right')

        port_2_idx[component + '-left'] = len(port_2_idx)

        port_2_idx[component + '-right'] = len(port_2_idx)

        comp2port_mapping[idx_component_in_pool] = [port_2_idx[component + '-left'], port_2_idx[component + '-right']]

        port2comp_mapping[port_2_idx[component + '-left']] = idx_component_in_pool

        port2comp_mapping[port_2_idx[component + '-right']] = idx_component_in_pool

        idx_2_port[len(idx_2_port)] = component + '-left'

        idx_2_port[len(idx_2_port)] = component + '-right'

        same_device_mapping[port_2_idx[component + '-left']] = port_2_idx[component + '-right']

        same_device_mapping[port_2_idx[component + '-right']] = port_2_idx[component + '-left']

    parent = list(range(len(port_pool)))

    return component_pool, port_pool, count_map, comp2port_mapping, port2comp_mapping, port_2_idx, idx_2_port, same_device_mapping, graph, parent


def convert_to_netlist(graph, component_pool, port_pool, parent, comp2port_mapping):
    list_of_node = set()

    list_of_edge = set()

    # netlist = []

    for idx, comp in enumerate(component_pool):
        # cur = []
        # cur.append(comp)
        list_of_node.add(comp)
        for port in comp2port_mapping[idx]:
            port_joint_set_root = find(port, parent)
            # cur.append(port_pool[port_joint_set_root])
            if port_joint_set_root in [0, 1, 2]:
                list_of_node.add(port_pool[port_joint_set_root])
                # list_of_edge.add((comp, port_pool[port_root]))
                list_of_node.add(port_joint_set_root)
                list_of_edge.add((comp, port_joint_set_root))
                list_of_edge.add((port_pool[port_joint_set_root], port_joint_set_root))
            else:
                list_of_node.add(port_joint_set_root)
                list_of_edge.add((comp, port_joint_set_root))

    netlist = []
    joint_list = set()

    for idx, comp in enumerate(component_pool):
        if comp in ['VIN', 'VOUT', 'GND']:
            continue
        cur = []

        cur.append(comp)
        for port in comp2port_mapping[idx]:
            # print(port_joint_set_root)
            port_joint_set_root = find(port, parent)
            root_0 = find(0, parent)
            root_1 = find(1, parent)
            root_2 = find(2, parent)
            if port_joint_set_root == root_0:
                cur.append("0")
            elif port_joint_set_root == root_1:
                cur.append("IN")

            elif port_joint_set_root == root_2:
                cur.append("OUT")
                # cur.append(port_pool[port_joint_set_root])
            # else:
            else:
                joint_list.add(str(port_joint_set_root))
                cur.append(str(port_joint_set_root))
        netlist.append(cur)

    return list(list_of_node), list(list_of_edge), netlist, list(joint_list)


def convert_graph(graph, comp2port_mapping, port2comp_mapping, idx_2_port, parent, component_pool, same_device_mapping,
                  port_pool):
    list_of_node = set()

    list_of_edge = set()

    has_short_cut = False

    for node in comp2port_mapping:
        if len(comp2port_mapping[node]) == 2:
            list_of_node.add(comp2port_mapping[node][0])
            list_of_node.add(comp2port_mapping[node][1])
            list_of_edge.add((comp2port_mapping[node][1], comp2port_mapping[node][0]))

    for node in graph:
        root_node = find(node, parent)
        list_of_node.add(node)
        list_of_node.add(root_node)

        # cur_node_the_other_port_root = find(cur_node_the_other_port, parent)

        if node in same_device_mapping:
            cur_node_the_other_port = same_device_mapping[node]
            cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
            if cur_node_the_other_port_root == root_node:
                has_short_cut = True

        if root_node != node:
            list_of_edge.add((node, root_node))

        for nei in graph[node]:
            list_of_node.add(nei)
            if nei != root_node:
                list_of_edge.add((nei, root_node))

    return list(list_of_node), list(list_of_edge), has_short_cut


def find_paths_from_edges(node_list, edge_list):
    return TopoGraph(node_list=node_list, edge_list=edge_list).find_end_points_paths_as_str()


def find_paths_from_adj_matrix(node_list, adj_matrix):
    return TopoGraph(node_list=node_list, adj_matrix=adj_matrix).find_end_points_paths_as_str()


def check_redundant_loop(node_list, edge_list):
    """
    Return the new node list and edge list with redundant loops removed
    """
    topo_graph = TopoGraph(node_list=node_list, edge_list=edge_list)
    topo_graph_copy = TopoGraph(node_list=node_list, edge_list=edge_list)
    topo_graph_copy.eliminate_redundant_comps()
    if len(topo_graph.get_node_list()) == len(topo_graph_copy.get_node_list()):
        return 0
    else:
        return 1


def remove_redundant_loop_from_edges(node_list, edge_list):
    """
    Return the new node list and edge list with redundant loops removed
    """
    topo_graph = TopoGraph(node_list=node_list, edge_list=edge_list)
    topo_graph.eliminate_redundant_comps()

    new_node_list = topo_graph.get_node_list()
    new_edge_list = topo_graph.get_edge_list()
    return new_node_list, new_edge_list


def check_topo_path(path, prohibit_path):
    for item in path:
        if item in prohibit_path:
            return False

    return True


def save_topo(list_of_node, list_of_edge, topo_file):
    T = nx.Graph()
    T.add_nodes_from((list_of_node))
    T.add_edges_from(list_of_edge)
    # plt.figure(1)
    # nx.draw(G, with_labels=True)
    plt.figure()
    nx.draw(T, with_labels=True)

    plt.savefig(topo_file)
    T.clear()
    plt.close()


def find_same_designs(node_list, edge_list, net_list):
    str_node = str(node_list);

    num_comp = len(net_list)
    num_devtype = 5

    num_C = str_node.count('C')
    num_L = str_node.count('L')
    num_Sa = str_node.count('Sa')
    num_Sb = str_node.count('Sb')
    num_R = str_node.count('R')
    num_list = [num_C, num_L, num_Sa, num_Sb, num_R]
    name_list = ['C', 'L', 'Sa', 'Sb', 'R']

    num_comp = len(net_list)
    count_max = max(num_C, 1) * max(num_L, 1) * max(num_Sa, 1) * max(num_Sb, 1) * max(num_R, 1)
    component_list = []

    count = 0
    k = 1
    tmp = []

    tmp_C = []
    tmp_L = []
    tmp_Sa = []
    tmp_Sb = []
    tmp_R = []
    name_list = []

    for i in range(num_list[0]):
        tmp_C.append('C' + str(i))
        name_list.append('C' + str(i))

    for i in range(num_list[1]):
        tmp_L.append('L' + str(i))
        name_list.append('L' + str(i))

    for i in range(num_list[2]):
        tmp_Sa.append('Sa' + str(i))
        name_list.append('Sa' + str(i))

    for i in range(num_list[3]):
        tmp_Sb.append('Sb' + str(i))
        name_list.append('Sb' + str(i))

    for i in range(num_list[4]):
        tmp_R.append('R' + str(i))
        name_list.append('R' + str(i))

    C_all = list(it.permutations(tmp_C))
    L_all = list(it.permutations(tmp_L))
    Sa_all = list(it.permutations(tmp_Sa))
    Sb_all = list(it.permutations(tmp_Sb))
    R_all = list(it.permutations(tmp_R))

    comp_all = [C_all, L_all, Sa_all, Sb_all, R_all]

    comb_all = list(it.product(*comp_all))

    comb_all_list = []

    for i in range(len(comb_all)):
        tmp = []
        for j in range(num_devtype):
            tmp.append(list(comb_all[i][j]))
        tmp_flag = []
        for item in tmp:
            for element in item:
                tmp_flag.append(element)
        comb_all_list.append(tmp_flag)

    edge_list_first = []
    net_list_first = []

    for item in edge_list:
        edge_list_first.append(item[0])

    for item in net_list:
        net_list_first.append(item[0])

    index_edge = []
    index_net = []

    for item in name_list:
        indices = [index for index, element in enumerate(edge_list_first) if element == item]
        index_edge.append(indices)
        indices = [index for index, element in enumerate(net_list_first) if element == item]
        index_net.append(indices)

    edge_list_group = []

    net_list_group = []

    for i in range(len(comb_all)):
        edge_list_copy = edge_list[:]
        net_list_copy = net_list[:]
        for j in range(num_comp):
            edge_list_copy[index_edge[j][0]][0] = comb_all_list[i][j]
            edge_list_copy[index_edge[j][1]][0] = comb_all_list[i][j]
            net_list_copy[index_net[j][0]][0] = comb_all_list[i][j]
        tmp = edge_list_copy[:]
        edge_list_group.append(str(edge_list_copy))
        net_list_group.append(str(net_list_copy))

    #    print(edge_list_group)

    return edge_list_group, net_list_group


def key_circuit_from_lists(edge_list, node_list, net_list):
    path = find_paths_from_edges(node_list, edge_list)

    node_dic = {}
    node_name = {}
    net_list_dic = {}

    for edge in edge_list:

        edge_start = edge[0]
        edge_end = edge[1]

        if edge_end in node_dic:
            node_dic[edge_end].append(edge_start)
        else:
            node_dic[edge_end] = []
            node_dic[edge_end].append(edge_start)

    for node in node_dic:

        node_dic[node].sort()
        name = 'N'

        for comp in node_dic[node]:
            name = name + '-' + comp

        if node in node_name:
            print('error')
        else:
            node_name[str(node)] = name

    tmp = net_list
    for item in tmp:

        for index, node in enumerate(item):
            if node == '0':
                item[index] = '0'
            elif node in node_name:
                item[index] = node_name[node]
        net_list_dic[item[0]] = item[1::]
        net_list_dic[item[0]].sort()

    net_list_dic_sorted = OrderedDict(sorted(net_list_dic.items()))

    key = str(net_list_dic_sorted)

    return key

def key_circuit_for_single_topo(edge_list_first, node_list, net_list_first):
    edge_list_group, net_list_group = find_same_designs(node_list, edge_list_first, net_list_first)
    key_list = []
    for kk in range(len(edge_list_group)):
        edge_list = literal_eval(edge_list_group[kk])
        net_list = literal_eval(net_list_group[kk])
        node_dic = {}
        node_name = {}
        net_list_dic = {}

        for edge in edge_list:
            edge_start = edge[0]
            edge_end = edge[1]
            if edge_end in node_dic:
                node_dic[edge_end].append(edge_start)
            else:
                node_dic[edge_end] = []
                node_dic[edge_end].append(edge_start)
        for node in node_dic:
            node_dic[node].sort()
            name = 'N'
            for comp in node_dic[node]:
                name = name + '-' + comp
            if node in node_name:
                print('error')
            else:
                node_name[str(node)] = name
        tmp = net_list
        for item in tmp:

            for index, node in enumerate(item):
                if node == '0':
                    item[index] = '0'
                elif node in node_name:
                    item[index] = node_name[node]
            net_list_dic[item[0]] = item[1::]
            net_list_dic[item[0]].sort()

        net_list_dic_sorted = OrderedDict(sorted(net_list_dic.items()))

        key = str(net_list_dic_sorted)
        key_list.append(key)

    return key_list


def key_circuit(fn, json_file):
    edge_list_first = json_file[fn]['list_of_edge']
    node_list = json_file[fn]['list_of_node']
    path = find_paths_from_edges(node_list, edge_list_first)

    net_list_first = json_file[fn]['netlist']
    edge_list_group = []
    net_list_group = []

    print("\n", fn)
    edge_list_group, net_list_group = find_same_designs(node_list, edge_list_first, net_list_first)

    key_list = []

    for kk in range(len(edge_list_group)):
        edge_list = literal_eval(edge_list_group[kk])
        net_list = literal_eval(net_list_group[kk])

        node_dic = {}
        node_name = {}
        net_list_dic = {}

        # print(net_list)
        for edge in edge_list:
            edge_start = edge[0]
            edge_end = edge[1]
            if edge_end in node_dic:
                node_dic[edge_end].append(edge_start)
            else:
                node_dic[edge_end] = []
                node_dic[edge_end].append(edge_start)
        for node in node_dic:
            node_dic[node].sort()
            name = 'N'
            for comp in node_dic[node]:
                name = name + '-' + comp
            if node in node_name:
                print('error')
            else:
                node_name[str(node)] = name
        tmp = net_list
        for item in tmp:

            for index, node in enumerate(item):
                if node == '0':
                    item[index] = '0'
                elif node in node_name:
                    item[index] = node_name[node]
            net_list_dic[item[0]] = item[1::]
            net_list_dic[item[0]].sort()

        net_list_dic_sorted = OrderedDict(sorted(net_list_dic.items()))

        key = str(net_list_dic_sorted)
        key_list.append(key)

    return key_list


def gen_param(device_list, parameters):
    param2sweep = {}
    param2sweep['Duty_Cycle'] = parameters['Duty_Cycle']
    param2sweep['Frequency'] = parameters['Frequency']
    param2sweep['Vin'] = parameters['Vin']
    param2sweep['Rout'] = parameters['Rout']
    param2sweep['Cout'] = parameters['Cout']
    param2sweep['Rin'] = parameters['Rin']

    for index, item in enumerate(device_list):

        if item[0] == 'R':
            if item[1] == 'a':
                param2sweep[item] = parameters['Ra']
            elif item[1] == 'b':
                param2sweep[item] = parameters['Rb']
            else:
                if item[1] != 'o' and item[1] != 'i':
                    param2sweep[item] = parameters['R']
        elif item[0] == 'C' and item[1] != 'o':
            param2sweep[item] = parameters['C']
        elif item[0] == 'L':
            param2sweep[item] = parameters['L']

    paramname = []
    for name in param2sweep:
        paramname.append(name)

    return param2sweep, paramname


def exp_subs(function, device, value):
    function_p = function.replace(device, value)
    return function_p


def convert_cki(fn, pv, dn, nt):
    path = './database/cki/' + fn
    file = open(path, 'w')

    print(dn)

    if 'Ra0' in dn:
        Ron = pv[dn['Ra0']]
        Roff = pv[dn['Rb0']]
    else:
        Ron = 1
        Roff = 100000

    prefix = [
        ".title buck.cki",
        ".model MOSN NMOS level=8 version=3.3.0",
        ".model MOSP PMOS level=8 version=3.3.0",
        ".model MySwitch SW (Ron=%s Roff=%s vt=%s)" % (Ron, Roff, pv[dn['Vin']] / 2),
        ".PARAM vin=%s rin=%s rout=%s cout=%su freq=%sM D=%s" % (
            pv[dn['Vin']], pv[dn['Rin']], pv[dn['Rout']], pv[dn['Cout']], pv[dn['Frequency']], pv[dn['Duty_Cycle']]),
        "\n",
        "*input*",
        "Vclock1 gate_a 0 PULSE (0 {vin} 0 1n 1n %su %su)" % (
            1 / pv[dn['Frequency']] * pv[dn['Duty_Cycle']], 1 / pv[dn['Frequency']]),
        "Vclock2 gate_b 0 PULSE ({vin} 0 0 1n 1n %su %su)" % (
            1 / pv[dn['Frequency']] * pv[dn['Duty_Cycle']], 1 / pv[dn['Frequency']]),

        "Vin IN_exact 0 dc {vin} ac 1",
        "Rin IN_exact IN {rin}",
        "Rout OUT 0 {rout}",
        "Cout OUT 0 {cout}"
        "\n"]

    sufix = ["\n",
             ".save all",
             # ".save i(vind)",
             ".control",
             # "tran %su 4000u" %(1/pv[dn['Frequency']]/10),
             "tran 10n 4000u",
             "print V(OUT)",
             "print V(IN_exact,IN)",
             ".endc",
             ".end",
             ]

    file.write("\n".join(prefix) + '\n')
    file.write("*topology*" + '\n')

    line = ''
    for x in nt:
        if 'S' == x[0][0]:
            if 'a' in x[0]:
                line = x[0] + ' ' + x[1] + ' ' + x[2] + ' gate_a gate_b MySwitch'
            elif 'b' in x[0]:
                line = x[0] + ' ' + x[1] + ' ' + x[2] + ' gate_b gate_a MySwitch'
        elif x[0][0] == 'C' or x[0][0] == 'L':
            line = x[0] + ' ' + x[1] + ' ' + x[2] + ' ' + str(pv[dn[x[0]]]) + 'u'
        elif x[0][0] == 'R':
            line = x[0] + ' ' + x[1] + ' ' + x[2] + ' ' + str(pv[dn[x[0]]])
        else:
            return 0

        line = line + '\n'
        file.write(line)

    file.write("\n".join(sufix) + '\n')
    file.close()
    return


def simulate(path):
    my_timeout = 30
    simu_file = path[:-3] + 'simu'
    p = subprocess.Popen("exec " + 'ngspice -b ' + path + '>' + simu_file, stdout=subprocess.PIPE, shell=True)
    try:
        p.wait(my_timeout)
    except subprocess.TimeoutExpired:
        print("kill\n")
        p.kill()


def check_switch(path):
    for item in path:
        if 'Sa' in item or 'Sb' in item:
            return 1
    return 0


def calculate_efficiency(path, input_voltage, freq, rin, rout):
    print(rin, rout)
    simu_file = path
    file = open(simu_file, 'r')
    V_in = input_voltage
    V_out = []
    I_in = []
    I_out = []
    time = []

    stable_ratio = 0.01

    cycle = 1 / freq
    # count = 0

    read_V_out, read_I_out, read_I_in = False, False, False
    for line in file:
        # print(line)
        if "Transient solution failed" in line:
            return {'result_valid': False,
                    'efficiency': -1,
                    'Vout': -1,
                    'Iin': -1,
                    'error_msg': 'transient_simulation_failure'}
        if "Index   time            v(out)" in line and not read_V_out:
            read_V_out = True
            read_I_in = False
            continue
        elif "Index   time            v(in_exact,in)" in line and not read_I_in:
            read_V_out = False
            read_I_in = True
            continue

        tokens = line.split()

        # print(tokens)
        if len(tokens) == 3 and tokens[0] != "Index":
            if read_V_out:
                time.append(float(tokens[1]))
                try:
                    V_out.append(float(tokens[2]))
                    I_out.append(float(tokens[2]) / rout)
                except:
                    print('Vout token error')
            elif read_I_in:
                try:
                    I_in.append(float(tokens[2]) / rin)
                except:
                    print('Iin token error')

    print(len(V_out), len(I_in), len(I_out))

    # print(len(V_out),len(I_out),len(I_in),len(time))
    if len(V_out) == len(I_in) == len(I_out) == len(time):
        pass
    else:
        print("don't match")
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -1,
                'Iin': -1,
                'error_msg': 'output_is_not_aligned'}

    if not V_out or not I_in or not I_out:
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -1,
                'Iin': -1,
                'error_msg': 'missing_output_type'}

    # print(I_out, I_in)
    end = len(V_out) - 1
    start = len(V_out) - 1
    print(cycle, start)
    while start >= 0:
        if time[end] - time[start] >= 50 * cycle:
            break
        start -= 1

    if start == -1:
        print("duration less than one cycle")
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -1,
                'Iin': -1,
                'error_msg': 'less_than_one_cycle'}
    mid = int((start + end) / 2)

    # print(start, end,time[end] - time[start])
    P_in = sum([(I_in[x] + I_in[x + 1]) / 2 * (V_in + V_in) / 2 *
                (time[x + 1] - time[x])
                for x in range(start, end)]) / (time[end] - time[start])

    P_out = sum([(I_out[x] + I_out[x + 1]) / 2 * (V_out[x] + V_out[x + 1]) / 2 *
                 (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave = sum([(V_out[x] + V_out[x + 1]) / 2 * (time[x + 1] - time[x])
                     for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave_1 = np.average(V_out[start:mid])

    V_out_ave_2 = np.average(V_out[mid:end - 1])

    I_in_ave = sum([(I_out[x] + I_out[x + 1]) / 2 * (time[x + 1] - time[x])
                    for x in range(start, end)]) / (time[end] - time[start])

    V_std = np.std(V_out[start:end - 1])

    # print('P_out, P_in', P_out, P_in)
    # if P_in == 0:
    if P_in < 0.001 and P_in > -0.001:
        P_in = 0
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -1,
                'Iin': -1,
                'error_msg': 'power_in_is_zero'}
    if P_out < 0.001 and P_out > -0.001:
        P_out = 0

    stable_flag = (abs(V_out_ave_1 - V_out_ave_2) <= max(abs(V_out_ave * stable_ratio), V_in / 200))

    # stable_flag = 1;

    print(P_out, P_in)

    eff = P_out / (P_in + 0.01)
    Vout = V_out_ave;
    Iin = I_in_ave;

    result = {'result_valid': (0 <= eff <= 1) and stable_flag,
              'efficiency': eff,
              'Vout': Vout,
              'Iin': Iin,
              'error_msg': 'None'}

    flag_candidate = 0

    if stable_flag == 0:
        result['error_msg'] = 'output_has_not_settled'

    elif eff < 0:
        result['error_msg'] = 'efficiency_is_less_than_zero'

    elif eff > 1:
        result['error_msg'] = 'efficiency_is_greater_than_one'
    elif (V_out_ave < 0.7 * input_voltage or V_out_ave > 1.2 * input_voltage) and eff > 0.7:
        flag_candidate = 1
        print('Promising candidates')

    return result


def get_mag_phase(path, F):
    simu_file = path
    file = open(simu_file, 'r')
    mag = []
    degree = []
    freq = []

    # count = 0

    read_mag, read_degree = False, False
    for line in file:

        if "Index   frequency       vdb(out)" in line and not read_mag:
            read_mag = True
            continue
        elif "Index   frequency       vp(out)" in line and not read_degree:
            read_mag = False
            read_degree = True
            continue

        tokens = line.split()
        # print(tokens)
        if len(tokens) == 3 and tokens[0] != "Index":
            if read_mag:
                try:
                    mag.append(float(tokens[2]))
                except:
                    print('Error')
            elif read_degree:
                freq.append(float(tokens[1]))
                try:
                    degree.append(float(tokens[2]))
                except:
                    print('Error')

    print(len(mag), len(degree), len(freq))

    print(mag, degree, freq)
    # print(len(V_out),len(I_out),len(I_in),len(time))
    if len(mag) == len(degree) == len(freq) and len(mag) > 0:
        return {'mag': mag[freq.index(F)],
                'degree': degree[freq.index(float(F))]
                }
    else:
        print("don't match")
        return {'mag': -120,
                'degree': 0}

    return result
