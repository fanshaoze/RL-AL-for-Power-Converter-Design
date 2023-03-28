import datetime
import warnings

from SimulatorAnalysis import gen_topo
from SimulatorAnalysis.simulate_with_topology import *
from ucts import uct
from utils.eliminate_isomorphism import sort_components
from utils.util import get_component_type, \
    get_component_count_suffix


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


def sort_dict_string(graph_dict):
    """
    Change the graph dict to an string, with sorted keys and sorted set value
    :graph_dict: the dict representing the graph
    """
    graph_dict_str = ""
    keys = graph_dict.keys()
    keys = list(keys)
    keys.sort()
    for key in keys:
        act_list = list(graph_dict[key])
        act_list.sort()
        graph_dict_str += str(key) + ":" + str(act_list) + ","
    return graph_dict_str


def get_observed_topology_count(graph_2_reward):
    """

    @param graph_2_reward: key + '$' + str(self.current.parameters):[D_C_L, eff, vout]
    @return: number of circuit topology in graph 2 reward
    """
    key_list = []
    for k, v in graph_2_reward.items():
        key = k.split('$')[0]
        if key not in key_list:
            key_list.append(key)
    return len(key_list)


def convert_graph_without_parents(graph):
    tmp_graph = deepcopy(graph)
    list_of_edge = set()

    for node in tmp_graph.keys():
        nei_set = tmp_graph[node]
        length_of_nei = len(nei_set)
        for _ in range(length_of_nei):
            list_of_edge.add((node, nei_set.pop()))

    return list(list_of_edge)


def find_connected_set(x, parent):
    net_list = []
    for i in range(len(parent)):
        if already_connected(x, i, parent):
            net_list.append(i)
    return net_list


def already_connected(x, y, parent):
    return find(x, parent) == find(y, parent)


def get_component_type(component):
    if component.startswith('L'):
        ret = 'L'
    elif component.startswith('C'):
        ret = 'C'
    elif component.startswith('Sa'):
        ret = 'Sa'
    elif component.startswith('Sb'):
        ret = 'Sb'
    else:
        ret = component
    return ret


def graph_has_short_cut(graph, parent, same_device_mapping):
    for node in graph:
        root_node = find(node, parent)
        if node in same_device_mapping:
            cur_node_the_other_port = same_device_mapping[node]
            cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
            if cur_node_the_other_port_root == root_node:
                return True
    return False


def graph_connect_roots(parent):
    gnd_root = find(0, parent)
    vin_root = find(1, parent)
    vout_root = find(2, parent)
    if gnd_root == vin_root or vin_root == vout_root or vout_root == gnd_root:
        return True
    return False


def convert_graph(graph, comp2port_mapping, parent, same_device_mapping, port_pool):
    list_of_node = set()
    list_of_edge = set()
    has_short_cut = False

    for node in comp2port_mapping:
        if len(comp2port_mapping[node]) == 2:
            list_of_node.add(port_pool[comp2port_mapping[node][0]])
            list_of_node.add(port_pool[comp2port_mapping[node][1]])
            list_of_edge.add((port_pool[comp2port_mapping[node][1]], port_pool[comp2port_mapping[node][0]]))

    for node in graph:
        root_node = find(node, parent)
        list_of_node.add(port_pool[node])
        list_of_node.add(port_pool[root_node])
        # TODO only for one device's short cut, but global case may not be achieved
        if node in same_device_mapping:
            cur_node_the_other_port = same_device_mapping[node]
            cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
            if cur_node_the_other_port_root == root_node:
                has_short_cut = True

        if root_node != node:
            list_of_edge.add((port_pool[node], port_pool[root_node]))

        for nei in graph[node]:
            list_of_node.add(port_pool[nei])
            if nei != root_node:
                list_of_edge.add((port_pool[nei], port_pool[root_node]))

    return list(list_of_node), list(list_of_edge), has_short_cut


def convert_to_netlist(component_pool, port_pool, parent, comp2port_mapping):
    list_of_node = set()
    list_of_edge = set()

    for idx, comp in enumerate(component_pool):
        list_of_node.add(comp)
        for port in comp2port_mapping[idx]:
            port_root = find(port, parent)
            if port_root in [0, 1, 2]:
                list_of_node.add(port_pool[port_root])
                list_of_node.add(port_root)
                list_of_edge.add((comp, port_root))
                list_of_edge.add((port_pool[port_root], port_root))
            else:
                list_of_node.add(port_root)
                list_of_edge.add((comp, port_root))

    return list(list_of_node), list(list_of_edge)


def remove_roots(allowed_root_pair, current_root_pair):
    for root_pair in allowed_root_pair:
        if (current_root_pair[0] in root_pair) and (current_root_pair[1] in root_pair):
            allowed_root_pair.remove(root_pair)
            break
    return allowed_root_pair


def find_roots(allowed_root_pair, current_root_pair):
    for root_pair in allowed_root_pair:
        if (current_root_pair[0] in root_pair) and (current_root_pair[1] in root_pair):
            random_prob = random.random()
            if random_prob <= allowed_root_pair[root_pair]:
                return True
            else:
                return False
    return False


#
class TopoGenState(uct.State):
    def __init__(self, init=False):
        if init:
            self.num_component = 0
            # self.component_pool = ['VIN', 'VOUT', "GND"]
            # self.port_pool = ['VIN', 'VOUT', "GND"]
            self.component_pool = ['GND', 'VIN', "VOUT"]
            self.port_pool = ['GND', 'VIN', "VOUT"]
            self.count_map = {"Sa": 0, "Sb": 0, "C": 0, "L": 0}
            self.comp2port_mapping = {0: [0], 1: [1],
                                      2: [2]}  # key is the idx in component pool, value is idx in port pool
            self.port2comp_mapping = {0: 0, 1: 1, 2: 2}

            self.port_2_idx = {'GND': 0, 'VIN': 1, "VOUT": 2}
            self.idx_2_port = {0: 'GND', 1: 'VIN', 2: "VOUT"}
            self.same_device_mapping = {}
            self.graph = collections.defaultdict(set)
            self.parent = None
            self.step = 0
            self.parameters = []

    # self.act_vect = []

    def init_disjoint_set(self):
        """
        The topology also use a union-find set to store the merged points, this is the init
        of union-find set printing the set and having a look(together with the graph dict)
        would be helpful
        """
        self.parent = list(range(len(self.port_pool)))

    def equal(self, state):
        if isinstance(state, TopoGenState):
            return self.component_pool == state.component_pool and \
                   self.port_pool == state.port_pool and \
                   self.num_component == state.num_component and \
                   self.count_map == state.count_map and \
                   self.comp2port_mapping == state.comp2port_mapping and \
                   self.port2comp_mapping == state.port2comp_mapping and \
                   self.port_2_idx == state.port_2_idx and \
                   self.idx_2_port == state.idx_2_port and \
                   self.same_device_mapping == state.same_device_mapping and \
                   self.graph == state.graph and \
                   self.step == state.step and \
                   self.parent == state.parent
        return False

    def get_edges(self):
        edges = []
        for key, vals in self.graph.items():
            for v in vals:
                edges.append((self.idx_2_port[key], self.idx_2_port[v]))
        return edges

    def duplicate(self):
        return deepcopy(self)

    def print(self):
        print('component_pool: {} \nport_pool: {}\nstep: {}'.format(self.component_pool, self.port_pool, self.step))

    def get_node_num(self):
        return len(self.component_pool) - len(['GND', 'VIN', 'VOUT'])

    def get_edge_num(self):
        edge_num = 0
        for key, val in self.graph.items():
            edge_num += len(val)
        return edge_num / 2

    def check_have_no_GND_path(self):
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(self.graph, self.component_pool, self.port_pool,
                                        self.parent, self.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge, True)
        gnd_root = find(self.port_2_idx['GND'], self.parent)
        for path in paths:
            components = path.split(' - ')
            if find_roots({('VIN', 'VOUT'): 1}, [components[0], components[-1]]):
                find_flag = True
                for i in range(1, len(components) - 1):
                    component = components[i]
                    left_port = component + '-left'
                    right_port = component + '-right'
                    if find(self.port_2_idx[left_port], self.parent) == gnd_root or \
                            find(self.port_2_idx[right_port], self.parent) == gnd_root:
                        # means this path has gnd on it, we need to search the next path
                        find_flag = False
                        break
                if find_flag:
                    # only in the case that we find a vin-vout path and on this path we do not meet a gnd that
                    # we can reach here. So we can return that,
                    # we find a valid graph which has a vin-vout path that do not go over gnd
                    return True
        return False

    def graph_is_connected(self):
        if self.graph:
            list_of_node, list_of_edge, has_short_cut = convert_graph(self.graph, self.comp2port_mapping, self.parent,
                                                                      self.same_device_mapping, self.port_pool)
            G = nx.Graph()
            G.add_nodes_from(list_of_node)
            G.add_edges_from(list_of_edge)
            return nx.is_connected(G)
        else:
            return False

    def has_all_ports(self):
        for i in range(len(['GND', 'VIN', 'VOUT']), len(self.port_pool)):
            if i not in self.graph:
                return False
        return True

    def has_in_out_gnd(self):
        return (1 in self.graph) and (2 in self.graph) and (0 in self.graph)

    def has_switch(self):
        for comp in self.component_pool:
            if ('Sa' in comp) or ('Sb' in comp):
                return True
        return False

    def graph_is_valid(self):
        if self.graph:
            list_of_node, list_of_edge, has_short_cut = convert_graph(self.graph, self.comp2port_mapping, self.parent,
                                                                      self.same_device_mapping, self.port_pool)
            G = nx.Graph()
            G.add_nodes_from(list_of_node)
            G.add_edges_from(list_of_edge)
            return self.check_have_no_GND_path() and (not has_short_cut) and self.has_in_out_gnd() and \
                   self.has_all_ports() and nx.is_connected(G) and self.has_switch() and \
                   (not gen_topo.check_redundant_loop(node_list=list_of_node, edge_list=list_of_edge))
        else:
            return False

    def get_idx_graph(self):
        if self.graph:
            list_of_node = list(range(len(self.port_pool)))
            list_of_edge = convert_graph_without_parents(self.graph)
            G = nx.Graph()
            G.add_nodes_from(list_of_node)
            G.add_edges_from(list_of_edge)
            return G
        else:
            return None

    def has_short_cut(self):
        for node in self.graph:
            root_node = find(node, self.parent)

            if node in self.same_device_mapping:
                cur_node_the_other_port = self.same_device_mapping[node]
                cur_node_the_other_port_root = find(cur_node_the_other_port, self.parent)
                if cur_node_the_other_port_root == root_node:
                    return True
        return False

    def visualize(self, title=None, figure_folder=None):
        list_of_node, list_of_edge = convert_to_netlist(self.component_pool, self.port_pool, self.parent,
                                                        self.comp2port_mapping)
        T = nx.Graph()
        T.add_nodes_from(list_of_node)
        T.add_edges_from(list_of_edge)
        if bool(title):
            print('title', title)
            plt.title(title)
        nx.draw(T, with_labels=True)

        # plt.show()
        dt = datetime.datetime.now().strftime(figure_folder + '%Y-%m-%d-%H-%M-%S')
        # dt = datetime.datetime.now().strftime(figure_folder + title)
        plt.savefig(dt)
        plt.close()

    def get_nodes_and_edges(self):
        """
        :return: (list of nodes, list of edges) of this state
        """
        return convert_to_netlist(self.component_pool, self.port_pool, self.parent, self.comp2port_mapping)

    def get_key(self):
        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.graph,
                                                                                      self.component_pool,
                                                                                      self.port_pool,
                                                                                      self.parent,
                                                                                      self.comp2port_mapping)
        return key_circuit_from_lists(list_of_edge, list_of_node, netlist)


class TopoGenAction(uct.SimAction):
    def __init__(self, action_type, value):
        self.type = action_type
        self.value = value

    def duplicate(self):
        other = TopoGenAction(self.type, self.value)
        return other

    def print(self):
        print(' ({}, {})'.format(self.type, self.value))

    def equal(self, other):
        if isinstance(other, TopoGenAction):
            return other.type == self.type and other.value == self.value
        return False


def port_list_multiply(port_lists_0, port_lists_1):
    merge_port_list = []
    for port_list_0 in port_lists_0:
        for port_list_1 in port_lists_1:
            tmp_list_0 = deepcopy(port_list_0)
            tmp_list_0.extend(port_list_1)
            merge_port_list.append(tmp_list_0)
    return merge_port_list


def path_add_edges(port_lists_0, port_lists_1):
    merge_port_list = []
    for port_list_0 in port_lists_0:
        for port_list_1 in port_lists_1:
            tmp_list_0 = deepcopy(port_list_0)
            tmp_list_0.append(port_list_1)
            merge_port_list.append(tmp_list_0)
    return merge_port_list


def port_multiply(port_list_0, port_list_1):
    edge_list = []
    for port_0 in port_list_0:
        for port_1 in port_list_1:
            edge = [port_0, port_1] if port_0 < port_1 else [port_1, port_0]
            edge_list.append(edge)
    return edge_list


def edge_list_multiply(candidate_edge_list_0, candidate_edge_list_1):
    merge_edge_list = []
    for candidate_edge_0 in candidate_edge_list_0:
        for candidate_edge_1 in candidate_edge_list_1:
            tmp_list_0 = deepcopy(candidate_edge_0)
            tmp_list_0.extend(candidate_edge_1)
            merge_edge_list.append(tmp_list_0)
    return merge_edge_list


def find_action(action, action_set):
    if isinstance(action, TopoGenAction):
        for i in range(len(action_set)):
            tmp_action = action_set[i]
            if action.type == tmp_action.type and action.value == tmp_action.value:
                return True
        return False
    return None


def remove_one_action(action, action_set):
    if isinstance(action, TopoGenAction):
        for i in range(action_set):
            tmp_action = action_set(i)
            if action.type == tmp_action.type and action.value == tmp_action.value:
                return True
        return False
    return None


def generate_port_path_add(path_adding_cost):
    """
    find the min length path to add and the edge start with current port when adding this path
    :param path_adding_cost: [(path, edge)]
    :return:
    """
    candidate_edge_list = []
    max_add_length = -1
    for path_edge in path_adding_cost:
        if len(path_edge[0]) > max_add_length:
            candidate_edge_list.clear()
            candidate_edge_list.append(path_edge[1])
        elif len(path_edge[0]) == max_add_length:
            candidate_edge_list.append(path_edge[1])
    return random.choice(candidate_edge_list)


def get_state_key(state):
    list_of_node, list_of_edge, netlist, joint_list = \
        gen_topo.convert_to_netlist(state.graph,
                                    state.component_pool,
                                    state.port_pool,
                                    state.parent,
                                    state.comp2port_mapping)
    return key_circuit_from_lists(list_of_edge, list_of_node, netlist)


def get_candidate_params():
    param = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    return param['Duty_Cycle']


def get_candidate_DC_C_L():
    param = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    candidate_paras = []
    for duty_cycle in param['Duty_Cycle']:
        for cap in param['C']:
            for ind in param['L']:
                candidate_paras.append([duty_cycle, cap, ind])
    return candidate_paras


def generate_possible_port_lists(components_with_count):
    component_port_pairs = []
    for comp_idx in range(1, len(components_with_count) - 1):
        component_port_pairs.append([[components_with_count[comp_idx] + '-left',
                                      components_with_count[comp_idx] + '-right'],
                                     [components_with_count[comp_idx] + '-right',
                                      components_with_count[comp_idx] + '-left']])
    path_port_list = component_port_pairs[0]
    for i in range(1, len(component_port_pairs)):
        path_port_list = port_list_multiply(path_port_list, component_port_pairs[i])
    return path_port_list


class TopoGenSimulator(uct.Simulator):
    def __init__(self, _configs, _approved_path_freq, _component_condition_prob, _key_expression,
                 _set_count_mapping,
                 _component_priority, _key_sim_effi=None, _fix_para_set=None, _num_component=4):
        if _key_sim_effi is None:
            _key_sim_effi = {}
        self.fix_para_set = _fix_para_set
        self.necessary_components = ["Sa", "Sb"]
        self.basic_components = ["Sa", "Sb", "L", "C"]
        self.reward = 0
        self.effi = 0
        self.vout = -500
        self.current = TopoGenState(init=True)
        # current_max: {state:, reward:, para:}
        self.current_max = {'state': None, 'reward': -1, 'para': None}
        self.configs_ = _configs
        self.num_component_ = _num_component
        self.query_counter = 0
        self.hash_counter = 0
        self.anal_total_time = 0
        self.new_query_time = 0
        self.new_query_counter = 0
        self.graph_2_reward = {}
        self.no_isom_seen_state_list = []
        self.encode_graph_2_reward = {}
        self.key_expression = _key_expression
        # count the number of visit of key, key:[count,reward,valid]
        self.key_sta = {}
        self.key_sim_effi_ = _key_sim_effi
        self.prohibit_path = ['VIN - L - GND', 'VIN - L - VOUT', 'VOUT - L - GND',
                              'VIN - Sa - GND', 'VIN - Sb - GND', 'VOUT - Sa - GND', 'VOUT - Sb - GND',
                              'VIN - Sa - Sa - GND', 'VIN - Sb - Sb - GND', 'VOUT - Sa - Sa - GND',
                              'VOUT - Sb - Sb - GND',
                              'VIN - Sa - Sa - Sa - GND', 'VIN - Sb - Sb - Sb - GND',
                              'VOUT - Sa - Sa - Sa - GND', 'VOUT - Sb - Sb - Sb - GND']
        # self.candidate_params = get_candidate_params()
        self.candidate_params = get_candidate_DC_C_L()
        self.topk = []
        self.approved_path_freq = _approved_path_freq
        self.component_condition_prob_ = _component_condition_prob
        # weights are the sample weights of act_vect, same idx, same length
        self.act_vect = []
        self.weights = []
        self.real_count = {}

        self.set_count_mapping = _set_count_mapping
        self.component_priority = _component_priority
        self.next_candidate_components = list(sort_components(tuple(self.basic_components),
                                                              self.component_priority))
        self.current_candidate_components = []

        self.update_action_set()

    def set_state(self, _next_candidate_components=None, _current_candidate_components=None, state=None):
        self.next_candidate_components = deepcopy(_next_candidate_components)
        self.current_candidate_components = deepcopy(_current_candidate_components)
        self.current = state.duplicate()
        self.update_action_set()

    def get_state(self):
        return self.current

    def get_sub_topo_condition_count(self, sub_topo, candidate_comps):
        sub_topo_condition_count = 0
        component_types = []
        for component in sub_topo:
            if ('GND' not in component) and ('VIN' not in component) and ('VOUT' not in component):
                component_types.append(get_component_type(component))
        sorted_comps = list(sort_components(tuple(component_types), self.component_priority))
        for k, v in self.set_count_mapping.items():
            # sa,sb,L,
            sorted_comps_tmp = deepcopy(sorted_comps)
            forbiden_comps = set(self.basic_components).difference((set(candidate_comps)))
            tmp_k = deepcopy(list(k))
            not_find_flag = 0
            for sort_comp in sorted_comps_tmp:
                if sort_comp not in tmp_k:
                    not_find_flag = 1
                    break
                else:
                    tmp_k.remove(sort_comp)
            if not_find_flag:
                continue
            if not set(tmp_k).intersection(forbiden_comps):
                if {'Sa', 'Sb', 'L'}.issubset(set(k)):
                    sub_topo_condition_count += v
        return sub_topo_condition_count

    def get_sub_topo_count(self, sub_topo, candidate_comps):
        sub_topo_count = 0
        component_types = []
        for component in sub_topo:
            if ('GND' not in component) and ('VIN' not in component) and ('VOUT' not in component):
                component_types.append(get_component_type(component))
        sorted_comps = list(sort_components(tuple(component_types), self.component_priority))
        for k, v in self.set_count_mapping.items():
            # sa,sb,L,
            sorted_comps_tmp = deepcopy(sorted_comps)
            forbiden_comps = set(self.basic_components).difference((set(candidate_comps)))
            tmp_k = deepcopy(list(k))
            not_find_flag = 0
            for sort_comp in sorted_comps_tmp:
                if sort_comp not in tmp_k:
                    not_find_flag = 1
                    break
                else:
                    tmp_k.remove(sort_comp)
            if not_find_flag:
                continue
            if not set(tmp_k).intersection(forbiden_comps):
                sub_topo_count += v
        if sub_topo_count == 0:
            print('error@get_sub_topo_count')
        return sub_topo_count

    def get_action_weights(self, branch_num, prestate_comps, prestate_n_can, prestate_c_can,
                           next_comps, next_n_can, nex_c_can):
        if len(prestate_comps) == self.num_component_:
            return 1
        else:
            if branch_num == 1:
                return 1
            else:
                prestate_count = self.get_sub_topo_count(prestate_comps,
                                                         list(set(prestate_n_can).union(set(prestate_c_can))))
                next_state_count = self.get_sub_topo_count(next_comps, list(set(next_n_can).union(set(nex_c_can))))
                return (prestate_count / branch_num) / next_state_count

    def finish_node_set(self):
        self.current.init_disjoint_set()

    def add_node(self, node_id):
        count = str(self.current.count_map[self.basic_components[node_id]])
        self.current.count_map[self.basic_components[node_id]] += 1
        component = self.basic_components[node_id] + count
        self.current.component_pool.append(component)
        idx_component_in_pool = len(self.current.component_pool) - 1
        self.current.port_pool.append(component + '-left')
        self.current.port_pool.append(component + '-right')
        self.current.port_2_idx[component + '-left'] = len(self.current.port_2_idx)
        self.current.port_2_idx[component + '-right'] = len(self.current.port_2_idx)
        self.current.comp2port_mapping[idx_component_in_pool] = [self.current.port_2_idx[component + '-left'],
                                                                 self.current.port_2_idx[component + '-right']]
        self.current.port2comp_mapping[self.current.port_2_idx[component + '-left']] = idx_component_in_pool
        self.current.port2comp_mapping[self.current.port_2_idx[component + '-right']] = idx_component_in_pool
        self.current.idx_2_port[len(self.current.idx_2_port)] = component + '-left'
        self.current.idx_2_port[len(self.current.idx_2_port)] = component + '-right'
        self.current.same_device_mapping[self.current.port_2_idx[component + '-left']] = self.current.port_2_idx[
            component + '-right']
        self.current.same_device_mapping[self.current.port_2_idx[component + '-right']] = self.current.port_2_idx[
            component + '-left']
        self.current.num_component += 1

    def add_edge(self, edge):
        if edge[0] < 0:
            return
        self.current.graph[edge[0]].add(edge[1])
        self.current.graph[edge[1]].add(edge[0])
        union(edge[0], edge[1], self.current.parent)
        return

    def edge_lead_to_prohibit_path(self, edge):
        tmp_state = deepcopy(self.get_state())
        if edge[0] < 0:
            return False
        tmp_state.graph[edge[0]].add(edge[1])
        tmp_state.graph[edge[1]].add(edge[0])
        union(edge[0], edge[1], tmp_state.parent)
        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(tmp_state.graph,
                                                                                      tmp_state.component_pool,
                                                                                      tmp_state.port_pool,
                                                                                      tmp_state.parent,
                                                                                      tmp_state.comp2port_mapping)
        path = find_paths_from_edges(list_of_node, list_of_edge)
        check_result = check_topo_path(path, self.prohibit_path)
        return not check_result

    def check_prohibit_path(self):
        list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
                                                                                      self.current.component_pool,
                                                                                      self.current.port_pool,
                                                                                      self.current.parent,
                                                                                      self.current.comp2port_mapping)
        path = find_paths_from_edges(list_of_node, list_of_edge)
        check_result = check_topo_path(path, self.prohibit_path)
        return check_result

    def get_edge_weight(self, e1, e2):
        weight = 0
        component_e1_idx = self.current.port2comp_mapping[e1]
        component_e2_idx = self.current.port2comp_mapping[e2]
        component_e1 = self.current.component_pool[component_e1_idx]
        component_e2 = self.current.component_pool[component_e2_idx]
        edge = get_component_type(component_e1) + ' - ' + get_component_type(component_e2)
        for path_freq in self.approved_path_freq:
            if edge in path_freq[0]:
                weight += 1
        return weight

    def get_edge_weight_with_freq(self, e1, e2):
        weight = 0
        component_e1_idx = self.current.port2comp_mapping[e1]
        component_e2_idx = self.current.port2comp_mapping[e2]
        component_e1 = self.current.component_pool[component_e1_idx]
        component_e2 = self.current.component_pool[component_e2_idx]
        edge = get_component_type(component_e1) + " - " + get_component_type(component_e2)
        reversed_edge = get_component_type(component_e2) + " - " + get_component_type(component_e1)
        for path_freq in self.approved_path_freq:
            if (edge in path_freq[0]) or (reversed_edge in path_freq[0]):
                weight += path_freq[1]
        return weight

    def reach_component_number(self):
        if len(self.current.component_pool) < self.num_component_:
            return False
        return True

    def update_action_set(self):
        """
        After action, the topology is changed, we need to find out which edges we can connect
        """
        # add components first
        if len(self.current.component_pool) < self.num_component_:
            if not self.current_candidate_components:
                self.current_candidate_components = deepcopy(self.next_candidate_components)
                self.next_candidate_components = []
            if len(self.current.component_pool) + 1 == self.num_component_:
                self.act_vect = []
                for action in self.current_candidate_components:
                    action_idx = self.basic_components.index(action)
                    self.act_vect.append(TopoGenAction('node', action_idx))
                for action in self.next_candidate_components:
                    action_idx = self.basic_components.index(action)
                    self.act_vect.append(TopoGenAction('node', action_idx))
            elif len(self.current_candidate_components) == 1 and self.next_candidate_components == []:
                action = self.current_candidate_components[0]
                action_idx = self.basic_components.index(action)
                self.act_vect = [TopoGenAction('node', action_idx)]
            elif len(self.current_candidate_components) >= 1:
                action = self.current_candidate_components[0]
                action_idx = self.basic_components.index(action)
                self.act_vect = [TopoGenAction('node', action_idx), TopoGenAction('node', -1)]
            elif len(self.current_candidate_components) == 0 and len(self.next_candidate_components) == 0:
                print('may find error')
            random.shuffle(self.act_vect)
            # we update the weights according to the count of leaf
            self.weights.clear()
            for action in self.act_vect:
                sub_topo = self.get_next_component_pool(action)
                tmp_n_comps, tmp_c_comps = self.get_update_candidates(action)
                candidate_comps = list(set(tmp_n_comps).union(set(tmp_c_comps)))
                next_total_count = self.get_sub_topo_count(sub_topo, candidate_comps)
                self.weights.append(next_total_count)
        # finish adding component, but not finish adding edge
        elif not self.finish_topology():
            edge_weight_list = []
            self.act_vect.clear()
            if self.current.graph_is_valid():
                self.act_vect.append(TopoGenAction('terminal', 0))
            e1 = self.current.step - (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT']))

            e1 %= len(self.current.port_pool)
            # add ports on vin first
            if e1 == 0:
                e1 = 1
            elif e1 == 1:
                e1 = 2
            elif e1 == 2:
                e1 = 0
            # after skipping them
            connected_set = find_connected_set(e1, self.current.parent)
            # if connected_set != [e1], it means e1 has connected with some other ports,
            # so we can skip, else we can not skip as we only allowed small port to connect large port
            if connected_set != [e1]:
                self.act_vect.append(TopoGenAction('edge', [-1, -1]))
            # TODO to let the ground search first
            # all the available edge set with e1 as a node
            e2_pool = list(range(len(self.current.port_pool)))
            # random.shuffle(e2_pool)

            for e2 in e2_pool:
                # assume we can not let large port to connect small port
                if e1 >= e2:
                    continue
                # from the same device
                if e2 in self.current.same_device_mapping and \
                        e1 == self.current.same_device_mapping[e2]:
                    continue
                # existing edges
                if (e1 in self.current.graph and e2 in self.current.graph[e1]) or \
                        (e2 in self.current.graph and e1 in self.current.graph[e2]):
                    continue
                # disjoint set
                e1_root = find(e1, self.current.parent)
                e2_root = find(e2, self.current.parent)
                gnd_root = find(0, self.current.parent)
                vin_root = find(1, self.current.parent)
                vout_root = find(2, self.current.parent)
                special_roots = [vin_root, vout_root, gnd_root]

                if e1_root in special_roots and e2_root in special_roots:
                    continue
                if e1_root == e2_root:
                    continue

                if e1 in self.current.same_device_mapping:
                    e1_other_port = self.current.same_device_mapping[e1]
                    e1_other_port_root = find(e1_other_port, self.current.parent)
                    if e1_other_port_root == e2_root:
                        continue

                if e2 in self.current.same_device_mapping:
                    e2_other_port = self.current.same_device_mapping[e2]
                    e2_other_port_root = find(e2_other_port, self.current.parent)
                    if e2_other_port_root == e1_root:
                        continue
                if self.configs_['prohibit_path']:
                    if self.edge_lead_to_prohibit_path([e1, e2]):
                        continue
                if self.configs_['approve_path']:
                    edge_weight = {'edge': [e1, e2], 'weight': self.get_edge_weight_with_freq(e1, e2)}
                    insert_flag = None
                    for i in range(len(edge_weight_list)):
                        if edge_weight['weight'] > edge_weight_list[i]['weight']:
                            edge_weight_list.insert(i, edge_weight)
                            insert_flag = True
                            break
                    if not insert_flag:
                        edge_weight_list.append(edge_weight)
                # remove same device port
                if find_connected_set(e2, self.current.parent) == [e2] and \
                        find_connected_set(self.current.same_device_mapping[e2], self.current.parent) == \
                        [self.current.same_device_mapping[e2]]:
                    # not included in RS
                    e2_pool.remove(self.current.same_device_mapping[e2])
                self.act_vect.append(TopoGenAction('edge', [e1, e2]))
            # It means, we can not skip e1, but we do not have allowed port to connect, what we can do is terminal
            if not self.act_vect:
                print("has to be unconnected")
                self.act_vect.append(TopoGenAction('terminal', 0))

            random.shuffle(self.act_vect)
            if len(self.act_vect) == 1:
                self.weights = [1]
            else:
                self.weights = [1 for i in range(len(self.act_vect))]
                for i in range(len(self.act_vect)):
                    if self.act_vect[i].equal(TopoGenAction('edge', [-1, -1])):
                        self.weights[i] = (0.8 / (1 - 0.8)) * (len(self.act_vect) - 1)
        # finish adding component and edge, but not terminal, means no sweep, just add parameter actions
        elif not self.is_terminal():
            self.act_vect.clear()
            for candidate_para in self.candidate_params:
                self.act_vect.append(TopoGenAction('parameter', candidate_para))
        elif self.is_terminal():
            # terminal, just stop
            self.act_vect.append(TopoGenAction('terminal', 0))
            self.weights.clear()
            self.weights.append(1)
        # assert len(self.weights) == len(self.act_vect)
        return

    def get_update_candidates(self, _action):
        assert _action.type == 'node'
        tmp_current_candidate_components = deepcopy(self.current_candidate_components)
        tmp_next_candidate_components = deepcopy(self.next_candidate_components)
        tmp_current_candidate_components.pop(0)
        if _action.value != -1:
            # means at least two candidate components to consider,so we can not directly taking _action
            # until num_component
            if tmp_current_candidate_components != [] or tmp_next_candidate_components != []:
                tmp_next_candidate_components.append(self.basic_components[_action.value])
        return tmp_current_candidate_components, tmp_next_candidate_components

    def get_next_component_pool(self, _action):
        assert _action.type == 'node'
        sub_topo = deepcopy(self.current.component_pool)
        if _action.value != -1:
            # only one type of component to add, so we directly add up to num_components
            if len(self.current_candidate_components) == 1 and self.next_candidate_components == []:
                for _ in range(self.num_component_ - len(self.current.component_pool)):
                    sub_topo.append(self.basic_components[_action.value])
            else:
                sub_topo.append(self.basic_components[_action.value])
        return sub_topo

    def act(self, _action, want_reward=True):

        parent_reward = 0
        if want_reward:
            parent_reward = self.get_reward()

        if _action.type == 'node':
            if _action.value != -1:
                # only one type of component to add, so we directly add up to num_components
                if len(self.current_candidate_components) == 1 and self.next_candidate_components == []:
                    for _ in range(self.num_component_ - len(self.current.component_pool)):
                        self.add_node(_action.value)
                        self.current.step += 1
                else:
                    self.add_node(_action.value)
                    self.current.step += 1
            self.current_candidate_components, self.next_candidate_components = self.get_update_candidates(_action)
        elif _action.type == 'edge':
            self.add_edge(_action.value)
            self.current.step += 1
        elif _action.type == 'terminal':
            self.current.step = len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT']) + len(
                self.current.port_pool)
        elif _action.type == 'parameter':
            self.current.parameters = _action.value
            self.current.step += 1
        else:
            print('Error: Unsupported Action Type!')

        if len(self.current.component_pool) == self.num_component_ and \
                not bool(self.current.parent):
            self.finish_node_set()
        if self.configs_['prohibit_path']:
            if _action.type == 'edge':
                if not self.check_prohibit_path():
                    print("checked forbidden path, but still have, error!")
                assert self.check_prohibit_path()
        self.update_action_set()
        if want_reward:
            self.reward = self.get_reward()
            self.reward = self.reward - parent_reward
            return self.reward
        else:
            return None

    def classify_special_topologies(self, reward, tmp_effi, tmp_vout):
        i = 3
        while i <= 10:
            if (i - 1) / 10 < reward <= i / 10:
                self.current.visualize('effi: ' + str(tmp_effi) + ' vout: ' + str(tmp_vout),
                                       str(i - 1) + 'to' + str(i) + '_')
                break
            i += 1
        return

    def update_key_sta(self, key, reward, valid_info):
        if self.key_sta.__contains__(key):
            self.key_sta[key][0] += 1
        else:
            self.key_sta[key] = [1, reward, valid_info]

    # start get reward related functions

    def get_reward(self):
        if not self.configs_['sweep']:
            return self.get_no_sweep_reward()
        else:
            return self.get_sweep_reward_with_para()

    def get_no_sweep_reward(self):
        reward = -1
        if self.configs_['reward_method'] == 'analytics':
            reward, _, _, _ = self.get_reward_using_anal()
        elif self.configs_['reward_method'] == 'simulator':
            reward, _, _, _ = self.get_reward_using_sim()
        return reward

    def get_sweep_reward_with_para(self):
        tmp_max_para = []
        tmp_max_reward = -1
        for paras in self.candidate_params:
            self.current.parameters = paras
            if self.configs_['reward_method'] == 'analytics':
                reward, _, _, _ = self.get_reward_using_anal()
            elif self.configs_['reward_method'] == 'simulator':
                reward, _, _, _ = self.get_reward_using_sim()
            else:
                warnings.warn('wrong reward method')
                return 'ERROR'
            if tmp_max_reward < reward:
                tmp_max_reward = reward
                tmp_max_para = paras
        self.current.parameters = tmp_max_para

        return tmp_max_reward

    def update_real_count(self):
        component_types = []
        for component in self.current.component_pool:
            if ('GND' not in component) and ('VIN' not in component) and ('VOUT' not in component):
                component_types.append(get_component_type(component))
        sorted_comps = list(sort_components(tuple(component_types), self.component_priority))
        if tuple(sorted_comps) in self.real_count:
            self.real_count[tuple(sorted_comps)] += 1
        else:
            self.real_count[tuple(sorted_comps)] = 1

    def get_reward_using_anal(self):
        # if not terminal or not valid return 0
        if not self.is_terminal():
            # init as invalid
            # self.current.parameters = []
            self.reward, self.effi, self.vout, = 0, 0, -500
            return self.reward, self.effi, self.vout, self.current.parameters
        else:
            self.update_real_count()
        # key_sta:[count,reward,valid]
        key = self.current.get_key()
        if not self.current.graph_is_valid():
            self.reward, self.effi, self.vout = 0, 0, -500
            self.update_key_sta(key, 0, False)
            return self.reward, self.effi, self.vout, self.current.parameters

        # If already in hash table, get the reward and return
        if self.graph_2_reward.__contains__(key + '$' + str(self.current.parameters)):
            efficiency = self.graph_2_reward[key + '$' + str(self.current.parameters)][1]
            vout = self.graph_2_reward[key + '$' + str(self.current.parameters)][2]
            tmp_reward = calculate_reward({'efficiency': efficiency, 'output_voltage': vout},
                                          self.configs_['target_vout'])

            self.reward, self.effi, self.vout = tmp_reward, efficiency, vout
            self.hash_counter += 1
            self.update_key_sta(key, tmp_reward, True)
        else:
            if self.configs_['skip_sim'] and \
                    (key + '$' + str(self.current.parameters) not in self.key_sim_effi_):
                print('skip_sim is true, no need to get as not in sim hash')
                tmp_reward, effi_info, tmp_effi, tmp_vout = 0, {'efficiency': 0, 'Vout': -500}, 0, 500
            else:
                tmp_reward, tmp_effi, tmp_vout = 0, 0, -500
                self.no_isom_seen_state_list.append([deepcopy(self.current), key])
                C_L_list_str = str([self.current.parameters[1], self.current.parameters[2]])
                if self.configs_['using_exp_inner_hash'] and \
                        (key + '$' + C_L_list_str in self.key_expression) and \
                        (str(self.current.parameters[0]) in self.key_expression[key + '$' + C_L_list_str]):
                    print('find analed topology %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    tmp_effi = self.key_expression[key + '$' + C_L_list_str][str(self.current.parameters[0])][
                        'Efficiency']
                    tmp_vout = self.key_expression[key + '$' + C_L_list_str][str(self.current.parameters[0])]['Vout']
                    tmp_reward = calculate_reward({'efficiency': tmp_effi, 'output_voltage': tmp_vout},
                                                  self.configs_['target_vout'])
                else:
                    start_time = time.time()
                    para_result, expression_time = \
                        get_one_analytics_result(current=self.current,
                                                 key_expression=self.key_expression,
                                                 graph=self.current.graph,
                                                 comp2port_mapping=self.current.comp2port_mapping,
                                                 port2comp_mapping=self.current.port2comp_mapping,
                                                 idx_2_port=self.current.idx_2_port,
                                                 port_2_idx=self.current.port_2_idx,
                                                 parent=self.current.parent,
                                                 component_pool=self.current.component_pool,
                                                 same_device_mapping=self.current.same_device_mapping,
                                                 port_pool=self.current.port_pool,
                                                 target_vout_min=self.configs_['min_vout'],
                                                 using_hash=self.configs_[
                                                     'using_exp_inner_hash'])
                    self.new_query_time += time.time() - start_time
                    self.new_query_counter += 1
                    if para_result:
                        assert str(self.current.parameters) in para_result
                        tmp_effi, tmp_vout = float(para_result[str(self.current.parameters)][0]), \
                                             float(para_result[str(self.current.parameters)][1])
                        tmp_reward = calculate_reward({'efficiency': tmp_effi, 'output_voltage': tmp_vout}
                                                      , self.configs_['target_vout'])

            self.graph_2_reward[key + '$' + str(self.current.parameters)] = [self.current.parameters, tmp_effi,
                                                                             tmp_vout]
            self.update_key_sta(key, tmp_reward, True)
            self.reward, self.effi, self.vout = tmp_reward, tmp_effi, tmp_vout
            self.query_counter += 1

            # No need to separate sweep or no-sweep in the max topo info, as max(no sweep) = max(sweep)+para
            self.update_max_topo_info()
            if self.configs_['sweep']:
                self.update_topk(key)
            else:
                self.update_topk_topology_with_para(key + '$' + str(self.current.parameters))

        return self.reward, self.effi, self.vout, self.current.parameters

    def get_reward_using_sim(self):
        if not self.is_terminal():
            self.current.parameters, self.reward, self.effi, self.vout = [], 0, 0, self.configs_['min_vout']
            return self.reward, self.effi, self.vout, self.current.parameters
        elif not self.current.graph_is_valid():
            self.current.parameters, self.reward, self.effi, self.vout = [], 0, 0, self.configs_['min_vout']
            # get key, the de isom string which to represent a sub topology
            key = self.current.get_key()
            # update the number of visit of key
            self.update_key_sta(key, 0, False)
            return self.reward, self.effi, self.vout, self.current.parameters
        key = self.current.get_key()
        # to isom, relpece graph_2_reward[key] with ...key and the next line
        if self.graph_2_reward.__contains__(key + '$' + str(self.current.parameters)):
            efficiency = self.graph_2_reward[key + '$' + str(self.current.parameters)][1]
            vout = self.graph_2_reward[key + '$' + str(self.current.parameters)][2]
            tmp_reward = calculate_reward({'efficiency': efficiency, 'output_voltage': vout},
                                          self.configs_['target_vout'])

            self.current.parameters = self.graph_2_reward[key + '$' + str(self.current.parameters)][0]
            self.reward = tmp_reward
            self.hash_counter += 1
        else:
            self.no_isom_seen_state_list.append([deepcopy(self.current), key])
            if key + '$' + str(self.current.parameters) in self.key_sim_effi_:
                print('find simulated topology %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                tmp_effi = self.key_sim_effi_[key + '$' + str(self.current.parameters)][0]
                tmp_vout = self.key_sim_effi_[key + '$' + str(self.current.parameters)][1]
                tmp_reward = calculate_reward({'efficiency': tmp_effi, 'output_voltage': tmp_vout},
                                              self.configs_['target_vout'])
                tmp_para = self.current.parameters
            else:
                if self.configs_['skip_sim']:
                    tmp_reward = 0
                    effi_info = {'efficiency': 0, 'Vout': 500}
                    tmp_para = self.current.parameters
                    tmp_effi = effi_info['efficiency']
                    tmp_vout = effi_info['Vout']
                else:
                    # if no sweep, we just get the reward of the para in current.parameter
                    start_time = time.time()
                    tmp_reward, tmp_effi, tmp_vout, tmp_para = \
                        get_single_topo_sim_result(current=self.current, sweep=False,
                                                   candidate_params=self.candidate_params,
                                                   key_sim_effi_=self.key_sim_effi_,
                                                   skip_sim=self.configs_['skip_sim'],
                                                   key_expression_mapping=self.key_expression,
                                                   target_vout=self.configs_['target_vout'],
                                                   min_vout=self.configs_['min_vout'])
                    # print("find_one_simulate_result using: ", str((datetime.datetime.now() - start_time).seconds))
                    self.new_query_time += time.time() - start_time
                    self.new_query_counter += 1
                    if not tmp_para:
                        tmp_reward, tmp_effi, tmp_vout = 0, 0, -500
                    self.key_sim_effi_[key + '$' + str(tmp_para)] = [tmp_effi, tmp_vout]
            self.graph_2_reward[key + '$' + str(tmp_para)] = [tmp_para, tmp_effi, tmp_vout]

            self.reward = tmp_reward
            self.effi = tmp_effi
            self.vout = tmp_vout
            self.query_counter += 1

            self.update_max_topo_info()
            if self.configs_['sweep']:
                self.update_topk(key)
            else:
                self.update_topk_topology_with_para(key + '$' + str(tmp_para))
        return self.reward, self.effi, self.vout, self.current.parameters

    def get_real_performance(self):
        if not self.is_terminal():
            self.current.parameters, self.reward, self.effi, self.vout = [], 0, 0, self.configs_['min_vout']
            return self.reward, self.effi, self.vout, self.current.parameters
        elif not self.current.graph_is_valid():
            self.current.parameters, self.reward, self.effi, self.vout = [], 0, 0, self.configs_['min_vout']
            return self.reward, self.effi, self.vout, self.current.parameters
        key = self.current.get_key()
        # to isom, relpece graph_2_reward[key] with ...key and the next line
        self.no_isom_seen_state_list.append([deepcopy(self.current), key])
        if key + '$' + str(self.current.parameters) in self.key_sim_effi_:
            print('find simulated topology %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            tmp_effi = self.key_sim_effi_[key + '$' + str(self.current.parameters)][0]
            tmp_vout = self.key_sim_effi_[key + '$' + str(self.current.parameters)][1]
            tmp_reward = calculate_reward({'efficiency': tmp_effi, 'output_voltage': tmp_vout},
                                          self.configs_['target_vout'])
        else:
            # if no sweep, we just get the reward of the para in current.parameter
            tmp_reward, tmp_effi, tmp_vout, tmp_para = \
                get_single_topo_sim_result(current=self.current, sweep=False,
                                           candidate_params=self.candidate_params,
                                           key_sim_effi_=self.key_sim_effi_,
                                           skip_sim=self.configs_['skip_sim'],
                                           key_expression_mapping=self.key_expression,
                                           target_vout=self.configs_['target_vout'],
                                           min_vout=self.configs_['min_vout'])
            # print("find_one_simulate_result using: ", str((datetime.datetime.now() - start_time).seconds))
            if not tmp_para:
                tmp_reward, tmp_effi, tmp_vout = 0, 0, -500
            self.key_sim_effi_[key + '$' + str(tmp_para)] = [tmp_effi, tmp_vout]
        self.reward = tmp_reward
        self.effi = tmp_effi
        self.vout = tmp_vout
        self.query_counter += 1

        return self.reward, self.effi, self.vout, self.current.parameters

    # end get reward related functions

    # start topk related function
    def update_topk(self, key_info):
        def keyFunc(element):
            return element[1]

        # De-isomorphism for topk
        if not self.topk_include_current(key_info):
            self.topk.append([self.current.duplicate(), self.reward, self.current.parameters, key_info])
            if len(self.topk) > self.configs_['topk_size']:
                self.topk.sort(key=keyFunc)
                self.topk.pop(0)
        else:
            for top_info_list in self.topk:
                # have the topo, but not the max reward para
                if (key_info == top_info_list[3]) and (self.reward > top_info_list[1]):
                    top_info_list[0], top_info_list[1], top_info_list[2] = \
                        self.current.duplicate(), self.reward, self.current.parameters
                    self.topk.sort(key=keyFunc)

    def update_topk_topology_with_para(self, key_info_with_para):
        def keyFunc(element):
            return element[1]

        # De-isomorphism for topk
        if not self.topk_include_current_with_para(key_info_with_para):
            key_info_list = key_info_with_para.split('$')
            key_info = key_info_list[0]
            self.topk.append([self.current.duplicate(), self.reward, self.current.parameters, key_info])
            if len(self.topk) > self.configs_['topk_size']:
                self.topk.sort(key=keyFunc)
                self.topk.pop(0)

    def update_max_topo_info(self):
        if (self.current_max == {}) or self.current_max['reward'] < self.reward:
            self.current_max['state'] = self.current.duplicate()
            self.current_max['reward'] = self.reward
            self.current_max['para'] = self.current.parameters

    def topk_include_current(self, key):
        for top_item in self.topk:
            if key == top_item[3]:
                return True
        return False

    def topk_include_current_with_para(self, key_info_with_para):
        for top_item in self.topk:
            # [self.current.duplicate(), self.reward, self.current.parameters, key_info]
            if key_info_with_para == top_item[3] + '$' + str(top_item[2]):
                return True
        return False

    def get_max_seen(self):
        return self.current_max['state'], self.current_max['reward'], self.current_max['para']

    # end topk related function

    def current_checking_port(self):
        e1 = (self.current.step - (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT'])))
        e1 %= len(self.current.port_pool)
        return e1

    def component_set_in_current(self, current, component_set):
        """
        add counts on current path's components and find whether the are exist in current sub graph
        :param component_set: components without count in path
        :return: components with count in path
        """
        count_set = {"Sa": 0, "Sb": 0, "C": 0, "L": 0}
        for idx in range(len(component_set)):
            if component_set[idx] in ['GND', 'VIN', 'VOUT']:
                continue
            else:
                count_set[component_set[idx]] += 1
                # in analytics, no - before count
                component_set[idx] = component_set[idx] + str(count_set[component_set[idx]] - 1)
        if set(component_set).issubset(set(current.component_pool)):
            return component_set
        else:
            return None

    def generate_primary_edges_to_add(self, path_port):
        edge_union_find_set = {}
        for i in range(0, len(path_port), 2):
            start = self.current.port_2_idx[path_port[i]]
            end = self.current.port_2_idx[path_port[i + 1]]
            edge = [start, end] if start < end else [end, start]
            if edge[0] in edge_union_find_set:
                edge_union_find_set[edge[0]].append(edge[1])
            else:
                edge_union_find_set[edge[0]] = [edge[1]]
        primary_edges_to_add = []
        for start, ports in edge_union_find_set.items():
            ports.sort()
            ports.insert(0, start)
            for i in range(len(ports) - 1):
                primary_edges_to_add.append([ports[i], ports[i + 1]])

        def keyFunc(element):
            return element[0]

        primary_edges_to_add.sort(key=keyFunc)
        return primary_edges_to_add

    def triangle_connection(self, edge):
        """check if A-B-C-A connection"""
        edge_0_other_port = self.current.same_device_mapping[edge[0]]
        edge_1_other_port = self.current.same_device_mapping[edge[1]]
        edge_0_root = find(edge[0], self.current.parent)
        edge_1_root = find(edge[1], self.current.parent)
        edge_0_other_port_root = find(edge_0_other_port, self.current.parent)
        edge_1_other_port_root = find(edge_1_other_port, self.current.parent)

        if (edge_0_other_port_root == edge_1_root) or \
                (edge_1_other_port_root == edge_0_root):
            return True
        return False

    def check_add_path(self, path_port, port_0, allowed_actions):
        """
        check if the topology is valid after adding the path
        """
        current_parent = deepcopy(self.current.parent)
        primary_edges_to_add = self.generate_primary_edges_to_add(path_port)
        path_possible_edges = [[]]
        for primary_edge in primary_edges_to_add:
            start = primary_edge[0]
            end = primary_edge[1]
            if already_connected(start, end, current_parent):
                continue
            net_of_start = find_connected_set(start, current_parent)
            net_of_end = find_connected_set(end, current_parent)
            possible_edges = port_multiply(net_of_start, net_of_end)
            i = 0
            check_root_connected_flag = False
            while i < len(possible_edges):
                edge = possible_edges[i]
                if not check_root_connected_flag:
                    e1_root = find(edge[0], self.current.parent)
                    e2_root = find(edge[1], self.current.parent)
                    gnd_root = find(0, self.current.parent)
                    vin_root = find(1, self.current.parent)
                    vout_root = find(2, self.current.parent)
                    special_roots = [vin_root, vout_root, gnd_root]
                    if e1_root in special_roots and e2_root in special_roots:
                        return None, []
                    else:
                        check_root_connected_flag = 1

                if edge[0] < port_0 or edge[1] < port_0:
                    possible_edges.remove(edge)
                    continue
                elif edge[0] in self.current.same_device_mapping and \
                        edge[1] in self.current.same_device_mapping:
                    if self.triangle_connection(edge):
                        possible_edges.remove(edge)
                        continue
                    else:
                        i += 1
                        continue
                else:
                    i += 1
                    continue
            if not possible_edges:
                return None, []
            path_possible_edges = path_add_edges(path_possible_edges, possible_edges)

        def keyFunc(element):
            return element[0]

        considerable_skip = []
        considerable_path_edge = []

        for possible_path in path_possible_edges:
            current_parent = deepcopy(self.current.parent)
            current_graph = deepcopy(self.current.graph)
            pre_start = -2
            need_add_list = []
            edge_to_add_current_port = []

            possible_path.sort(key=keyFunc)

            for edge in possible_path:
                start = edge[0]
                end = edge[1]
                # need add two ports on one edge
                if start == pre_start:
                    continue
                if not already_connected(start, end, current_parent):
                    # we need to add port on previous considered ports
                    if start < port_0 or end < port_0:
                        continue
                    else:
                        if start == port_0:
                            # we need to add some port on current considering port which is
                            # not allowed
                            if not find_action(TopoGenAction('edge', edge), allowed_actions):
                                continue
                            else:
                                edge_to_add_current_port = edge
                    need_add_list.append([start, end])
                    union(start, end, current_parent)
                    current_graph[start].add(end)
                    current_graph[end].add(start)
                pre_start = start
            if graph_has_short_cut(current_graph, current_parent, self.current.same_device_mapping):
                continue
            elif graph_connect_roots(current_parent):
                continue
            else:
                """we use a rule that for a path:
                    if: we both skip([-1,-1]) and adding an edge[port_0, port_1] is possible, we only 
                    consider the minimum number of adding edges for adding an edge
                    else, if we only have skip[-1,-1] for choise, we only consider skip, still minimum 
                    number of adding edges
                """
                if need_add_list:
                    if edge_to_add_current_port:
                        if (not considerable_path_edge) or \
                                len(need_add_list) < len(considerable_path_edge[0]):
                            considerable_path_edge = [need_add_list, edge_to_add_current_port]
                else:
                    continue
        if considerable_path_edge:
            return considerable_path_edge[0], considerable_path_edge[1]
        elif considerable_skip:
            return considerable_skip[0], considerable_skip[1]
        else:
            return None, []

    def edges_adding_from_path(self, allowed_root_pair, path, port_0, allowed_actions):
        """
        return the edges need to be add if we want to let this path generated in current sub-graph
        if adding this path lead to invalid topology, return [False, []]
        :param allowed_root_pair: root pair can be considered for path
        :param path: path we consider
        :param port_0: starting port
        :param allowed_actions: candidate edge list
        :return:
        """
        path = str(path)
        ''' 'VIN - Sa - L - GND' '''
        components = path.split(' - ')
        ''' components: [VIN, Sa, L, GND] '''
        components_with_count = self.component_set_in_current(current=self.current, component_set=components)
        ''' components_with_count: [VIN, Sa0, L0, GND] '''
        if not components_with_count:
            return False, []
        elif not find_roots(allowed_root_pair, [components[0], components[-1]]):
            return False, []
        else:
            """
            path port_list: 
                [Sa0-left,Sa0-right, L-0-left, L-0-right]
                [Sa0-right,Sa0-left, L-0-left, L-0-right]
                [Sa0-left,Sa0-right, L-0-right, L-0-left]
                [Sa0-right,Sa0-left, L-0-right, L-0-left]
                """
            path_port_list = generate_possible_port_lists(components_with_count)

        path_adding_cost = []
        for path_port in path_port_list:
            path_port.insert(0, components[0])
            path_port.append(components[-1])

            need_add_edge_list, edge_to_add_current_port = self.check_add_path(path_port,
                                                                               port_0,
                                                                               allowed_actions)
            if need_add_edge_list:
                if edge_to_add_current_port:
                    path_adding_cost.append((need_add_edge_list, edge_to_add_current_port))
            else:
                continue
        if path_adding_cost:
            edge_to_add = generate_port_path_add(path_adding_cost)
            return True, edge_to_add
        return False, []

    def generate_path_weight(self, allowed_root_pair, port_0, allowed_actions):
        """
        we get the path that allowed to be add in
        :param allowed_root_pair: the root pair that has not been connected in  current graph
        :param port_0: the start port that current round is considering
        :param allowed_actions: all the actions that allowed in this round(will not lead to shortcut and direct of roots
        :return:  path_with_probability: key: path's index(idx), value:[edge_to_add(from port_0), probability]
        """
        sum_freq = 0
        edge_with_probability = {}
        approved_path = []
        for path_freq in self.approved_path_freq:
            approved_path.append(path_freq[0])
        for path_idx in range(len(approved_path)):
            # can add: path with idx can be add or not? as adding some path may lead to invalid path
            # edge to add: the edge(port0, portk) that need to add if we want to generate path(idx)
            can_add, edge_to_add = self.edges_adding_from_path(allowed_root_pair, approved_path[path_idx],
                                                               port_0, allowed_actions)
            if can_add:
                edge_with_probability[path_idx] = edge_to_add
                # sum freq not used
                sum_freq += self.approved_path_freq[path_idx][1]
        # for actions in allowed_actions:
        for path_idx, edge_to_add in edge_with_probability.items():
            edge_with_probability[path_idx] = (edge_to_add, self.approved_path_freq[path_idx][1])
        return edge_with_probability

    def get_roots_of_path(self, path):
        """
        get the root pair of a path
        :param path: VIN-L-C-VOUT, string form
        :return: [VIN, VOUT]
        """
        root_0 = self.current.idx_2_port[path[0][0]]
        root_1 = self.current.idx_2_port[path[-1][-1]]
        return [root_0, root_1]

    def find_path_in_approved_path(self, path):
        for path_freq in self.approved_path_freq:
            if path == path_freq[0]:
                return True
        return False

    def find_not_connected_roots(self):
        """
        find the not connected root pairs, we just allow using approve path to connect one path between
        each pair of roots
        :return: list of root pair that not connected,
        """
        allowed_root_pair = [['VIN', 'VOUT'], ['VOUT', 'GND'], ['VIN', 'GND']]
        current_state = self.get_state()
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(current_state.graph,
                                        current_state.component_pool,
                                        current_state.port_pool,
                                        current_state.parent,
                                        current_state.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        for path in paths:
            # print(path)
            if self.find_path_in_approved_path(path):
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$built a good path:", path)
                path_comp = path.split(' - ')
                allowed_root_pair = remove_roots(allowed_root_pair, [path_comp[0], path_comp[-1]])
                if not allowed_root_pair:
                    return []
        return allowed_root_pair

    def generate_pair_prob(self):
        """
        generate pairs probs, the pairs that has already connected with good path has a lower prob,
        which will be less considered in the good-path selection process for generate the weights
        for edges
        :return: root pairs and prob
        """
        allowed_root_pair = {('VIN', 'VOUT'): 1, ('VOUT', 'GND'): 1, ('VIN', 'GND'): 1}
        current_state = self.get_state()
        list_of_node, list_of_edge, netlist, joint_list = \
            gen_topo.convert_to_netlist(current_state.graph,
                                        current_state.component_pool,
                                        current_state.port_pool,
                                        current_state.parent,
                                        current_state.comp2port_mapping)
        paths = gen_topo.find_paths_from_edges(list_of_node, list_of_edge)
        for path in paths:
            # print(path)
            if self.find_path_in_approved_path(path):
                path_comp = path.split(' - ')
                allowed_root_pair[(path_comp[0], path_comp[-1])] = 0.5
        return allowed_root_pair

    def sample_from_paths(self, paths_with_weights):
        assert paths_with_weights
        # we use edge, as the action form is [type, [port0, port1](edge)], easier to output
        # edge and edge's sum_weight, index-to-index
        edges = []
        edge_weights = []

        basic_weight = self.configs_['DP_path_basic_weight']
        sum_weights = 0
        # first the edges have related paths, then the not found related paths

        # paths_with_weights: path idx, [the edge that used generate path, prob]
        for path_idx, edge_and_prob in paths_with_weights.items():
            # path idx not use, the prob of path_idx's path is stored in edge_prob
            sum_weights += edge_and_prob[1]
            if edge_and_prob[0] in edges:
                edge_weights[edges.index(edge_and_prob[0])] += edge_and_prob[1]
            else:
                edges.append(edge_and_prob[0])
                edge_weights.append(edge_and_prob[1])
        for edge in edges:
            edge_weights[edges.index(edge)] = \
                basic_weight / len(self.get_actions()) + \
                (1 - basic_weight) * edge_weights[edges.index(edge)] / sum_weights
        for action in self.get_actions():
            if action.value not in edges:
                edges.append(action.value)
                edge_weights.append(basic_weight / len(self.get_actions()))
        if 0.0 in edge_weights:
            print()
        edge = random.choices(edges, weights=edge_weights, k=1)
        return edge[0]

    def sample_component(self):
        allowed_actions = deepcopy(self.act_vect)
        action_weights = []
        basic_weight = self.configs_['DP_component_basic_weight']
        sum_condition_count = 0

        sub_topo = self.current.component_pool
        candidate_comps = list(set(self.next_candidate_components).union(set(self.current_candidate_components)))

        current_condition_count = self.get_sub_topo_condition_count(sub_topo, candidate_comps)
        current_total_count = self.get_sub_topo_count(sub_topo, candidate_comps)
        if current_condition_count == 0:
            for action in allowed_actions:
                if action.type == 'edge':
                    print(1)

                sub_topo = self.get_next_component_pool(action)
                tmp_n_comps, tmp_c_comps = self.get_update_candidates(action)
                candidate_comps = list(set(tmp_n_comps).union(set(tmp_c_comps)))

                next_total_count = self.get_sub_topo_count(sub_topo, candidate_comps)
                action_weights.append(next_total_count / current_total_count)
        else:
            for action in allowed_actions:
                if action.type == 'edge':
                    print(1)
                sub_topo = self.get_next_component_pool(action)
                tmp_n_comps, tmp_c_comps = self.get_update_candidates(action)
                candidate_comps = list(set(tmp_n_comps).union(set(tmp_c_comps)))

                next_condition_count = self.get_sub_topo_condition_count(sub_topo, candidate_comps)
                next_total_count = self.get_sub_topo_count(sub_topo, candidate_comps)
                action_weights.append(basic_weight * next_total_count / current_total_count +
                                      (1 - basic_weight) * next_condition_count / current_condition_count)
                sum_condition_count += next_condition_count

            if sum_condition_count != current_condition_count:
                print(1)
        if action_weights:
            action_list = random.choices(allowed_actions, weights=action_weights, k=1)
            return action_list[0]
        else:
            return None

    def get_action_using_default_policy(self, allowed_actions, dp_for_component, dp_for_path):
        """
        get an action that (1) can be add in current topology according to approve path set(find a path which can be add
        into current topology, then return the edge which is (2) allowed in current round
        :param allowed_actions: the allowed action set in current round. for example, we want to add the edges one of
        whose end is 3, then all the allowed actions are [3. x] or [-1, -1]
        :dp_for_component: using dp to get component
        :dp_for_path: using dp to get edge(with path information)
        :return: action(edge, [port_0, port_1])
        """
        # print("----------------------------------")
        if not self.reach_component_number():
            if dp_for_component:
                action = self.sample_component()
            else:
                # act_ID = int(random.random() * len(allowed_actions))
                if len(self.act_vect) != len(self.weights):
                    print(len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
            return action
        elif not self.finish_topology():
            if dp_for_path:
                # The already connected root pair has a lower prob
                allowed_root_pair = self.generate_pair_prob()
                if not allowed_root_pair:
                    return None
                port_0 = self.current_checking_port()
                edges_with_weights = self.generate_path_weight(allowed_root_pair, port_0, allowed_actions)
                if not edges_with_weights:
                    return None
                edge_to_add = self.sample_from_paths(edges_with_weights)
                if edge_to_add == 0:
                    return TopoGenAction('terminal', 0)
                return TopoGenAction('edge', edge_to_add)
            else:
                if len(self.act_vect) != len(self.weights):
                    print('error of weight length------,', len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
                return action
        else:
            # print(allowed_actions)
            act_ID = int(random.random() * len(allowed_actions))
            action = allowed_actions[act_ID]
            # print(f'{act_ID} select duty cycle: {action.value}')
            return action

    def default_policy(self, mc_return, gamma, discount, reward_list, dp_for_component, dp_for_path):
        while not self.is_terminal():
            allowed_actions = deepcopy(self.get_actions())
            action = self.get_action_using_default_policy(allowed_actions, dp_for_component, dp_for_path)
            if action is None:
                if len(self.act_vect) != len(self.weights):
                    print(len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
            r = self.act(action)
            reward_list.append(r)
            mc_return += discount * r
            discount *= gamma
        return mc_return

    def generate_random_topology_without_reward(self):
        while not self.is_terminal():
            allowed_actions = deepcopy(self.get_actions())
            action = self.get_action_using_default_policy(allowed_actions, False, False)
            if action is None:
                if len(self.act_vect) != len(self.weights):
                    print(len(self.act_vect), len(self.weights))
                choose_act = random.choices(self.act_vect, weights=self.weights, k=1)
                action = choose_act[0]
            self.act(action, False)
        # rand_para = random.choice(self.candidate_params)
        # self.current.parameters = rand_para
        key = self.current.get_key()
        # list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
        #                                                                               self.current.component_pool,
        #                                                                               self.current.port_pool,
        #                                                                               self.current.parent,
        #                                                                               self.current.comp2port_mapping)
        # key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        return self.current, key

    # end the default policy

    def get_actions(self):
        return self.act_vect

    def get_next_candidate_components(self):
        return self.next_candidate_components

    def get_current_candidate_components(self):
        return self.current_candidate_components

    def set_next_candidate_components(self, _next_candidate_components):
        self.next_candidate_components = _next_candidate_components

    def get_weights(self):
        return self.weights

    def is_terminal(self):
        if self.configs_['sweep']:
            return self.finish_topology()
        else:
            return \
                self.current.step >= (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT'])) + \
                len(self.current.port_pool) + 1

    def finish_topology(self):
        if self.current.step >= (len(self.current.component_pool) - len(['GND', 'VIN', 'VOUT'])) + \
                len(self.current.port_pool):
            return True
        return False

    # start Genetic Search related functions

    def change_node(self, pre_component_id, changed_node_id):

        pre_component = self.current.component_pool[pre_component_id]
        pre_ports_idx = self.current.comp2port_mapping[pre_component_id]
        pre_left_port_index = pre_ports_idx[0]
        pre_right_port_index = pre_ports_idx[1]
        pre_component_type = get_component_type(pre_component)
        changed_component_type = self.basic_components[changed_node_id]
        if changed_component_type == pre_component_type:
            return
            # print(pre_component_type)
        self.current.count_map[pre_component_type] -= 1
        pre_component_suffix = get_component_count_suffix(pre_component)
        for i in range(len(self.current.component_pool)):
            component = self.current.component_pool[i]
            if i != pre_component_id and get_component_type(component) == pre_component_type:
                suffix = get_component_count_suffix(component)
                if suffix > pre_component_suffix:
                    self.current.component_pool[i] = get_component_type(component) + \
                                                     str(suffix - 1)

        for i in range(len(self.current.component_pool)):
            component = self.current.component_pool[i]
            if i != pre_component_id and get_component_type(component) == pre_component_type:
                suffix = get_component_count_suffix(component)
                if suffix > pre_component_suffix:
                    # replace component's suffix
                    self.current.component_pool[i] = get_component_type(component) + str(suffix - 1)
                    # replace ports' suffix
                    ports = self.current.comp2port_mapping[i]
                    self.current.port_pool[ports[0]] = self.current.port_pool[ports[0]].replace(
                        component, get_component_type(component) + str(suffix - 1))

        self.current.same_device_mapping.pop(self.current.port_2_idx[pre_component + '-left'])
        self.current.same_device_mapping.pop(self.current.port_2_idx[pre_component + '-right'])

        # changed_component = self.basic_components[changed_node_id] + '-' + str(pre_component_id)
        self.current.count_map[changed_component_type] += 1
        changed_component = self.basic_components[changed_node_id] + str(self.current.count_map[changed_component_type])

        self.current.component_pool[pre_component_id] = changed_component

        self.current.port_pool[pre_left_port_index] = changed_component + '-left'
        self.current.port_pool[pre_right_port_index] = changed_component + '-right'

        self.current.idx_2_port[pre_left_port_index] = changed_component + '-left'
        self.current.idx_2_port[pre_right_port_index] = changed_component + '-right'
        self.current.port_2_idx[changed_component + '-left'] = self.current.port_2_idx.pop(pre_component + '-left')
        self.current.port_2_idx[changed_component + '-right'] = self.current.port_2_idx.pop(pre_component + '-right')
        self.current.same_device_mapping[self.current.port_2_idx[changed_component + '-left']] = \
            self.current.port_2_idx[changed_component + '-right']
        self.current.same_device_mapping[self.current.port_2_idx[changed_component + '-right']] = \
            self.current.port_2_idx[changed_component + '-left']
        return

    def delete_edge(self, edge):
        # remove edge 0 and edge 1 from graph
        self.current.graph[edge[0]].remove(edge[1])
        self.current.graph[edge[1]].remove(edge[0])
        if not self.current.graph[edge[0]]:
            self.current.graph.pop(edge[0])
        if not self.current.graph[edge[1]]:
            self.current.graph.pop(edge[1])
        root_port = self.current.parent[edge[0]]
        # print("e0 and root,e2 and root", edge[0], self.current.parent[edge[0]], edge[1], self.current.parent[edge[1]])
        idx_graph = self.current.get_idx_graph()
        self.current.step -= 1
        # for i in range(len(self.current.parent)):
        #     print(i, nx.has_path(idx_graph, i, root_port))
        '''
        if 0, 1 and 2r(root port) are connected together, three possible ways(0-1 had an edge):
        0-1, 0-r: delete 0-1, 1 and r are not connected, so the ports connected with r through 1 change root to 1  
        0-1, 1-r: delete 0-1, 0 and r are not connected, so the ports connected with r through 0 change root to 0
        0-1, 0-r, 1-r: impossible, as in update_action_set we prohibit adding edge between indirectly connected ports
        '''
        if nx.has_path(idx_graph, edge[0], root_port):
            for i in range(len(self.current.parent)):
                if self.current.parent[i] == root_port:
                    if not nx.has_path(idx_graph, i, root_port):
                        self.current.parent[i] = edge[1]
        elif nx.has_path(idx_graph, edge[1], root_port):
            for i in range(len(self.current.parent)):
                if self.current.parent[i] == root_port:
                    if not nx.has_path(idx_graph, i, root_port):
                        self.current.parent[i] = edge[0]
        else:
            print("delete edge wrongly")

    def find_new_edges(self, e1):
        # print(e1)
        edge_weight_list = []
        self.act_vect.clear()
        if self.current.graph_is_valid():
            self.act_vect.append(TopoGenAction('terminal', 0))
        connected_set = find_connected_set(e1, self.current.parent)
        # if connected_set != [e1], it means e1 has connected with some other ports,
        # so we can skip, else we can not skip as we only allowed small port to connect large port
        if connected_set != [e1]:
            self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        # self.act_vect.append(TopoGenAction('edge', [-1, -1]))
        # TODO to let the ground search first
        # all the available edge set with e1 as a node
        e2_pool = list(range(len(self.current.port_pool)))
        random.shuffle(e2_pool)

        for e2 in e2_pool:
            # assume we can not let large port to connect small port
            if e1 >= e2:
                continue
            # from the same device
            if e2 in self.current.same_device_mapping and \
                    e1 == self.current.same_device_mapping[e2]:
                continue
            # existing edges
            if (e1 in self.current.graph and e2 in self.current.graph[e1]) or \
                    (e2 in self.current.graph and e1 in self.current.graph[e2]):
                continue
            # disjoint set
            e1_root = find(e1, self.current.parent)
            e2_root = find(e2, self.current.parent)
            # TODO fix the order, althouth currently not effect the result
            gnd_root = find(0, self.current.parent)
            vin_root = find(1, self.current.parent)
            vout_root = find(2, self.current.parent)
            special_roots = [vin_root, vout_root, gnd_root]

            if e1_root in special_roots and e2_root in special_roots:
                continue
            if e1_root == e2_root:
                continue

            if e1 in self.current.same_device_mapping:
                e1_other_port = self.current.same_device_mapping[e1]
                e1_other_port_root = find(e1_other_port, self.current.parent)
                if e1_other_port_root == e2_root:
                    continue

            if e2 in self.current.same_device_mapping:
                e2_other_port = self.current.same_device_mapping[e2]
                e2_other_port_root = find(e2_other_port, self.current.parent)
                if e2_other_port_root == e1_root:
                    continue
            if self.configs_['prohibit_path']:
                if self.edge_lead_to_prohibit_path([e1, e2]):
                    continue
            if self.configs_['approve_path']:
                edge_weight = {'edge': [e1, e2], 'weight': self.get_edge_weight_with_freq(e1, e2)}
                insert_flag = None
                for i in range(len(edge_weight_list)):
                    if edge_weight['weight'] > edge_weight_list[i]['weight']:
                        edge_weight_list.insert(i, edge_weight)
                        insert_flag = True
                        break
                if not insert_flag:
                    edge_weight_list.append(edge_weight)

            self.act_vect.append(TopoGenAction('edge', [e1, e2]))
        # It means, we can not skip e1, but we do not have allowed port to connect, what we can do is terminal
        if not self.act_vect:
            print("has to be unconnected")
        return

    def change_edge(self, e1, e2):
        self.delete_edge([e1, e2])
        print("Graph after delete edge", self.current.graph)
        print("parent after delete edge", self.current.parent)
        self.find_new_edges(e2)
        for i in range(len(self.act_vect)):
            self.act_vect[i].print()
        if len(self.act_vect) == 1:
            warnings.warn("The deleted edge is added into the graph again")
        action = random.choice(self.act_vect)
        print("selected action from adding the edge in mutation")
        r = self.act(action)
        return r

    def mutate(self, action_porbs=None):
        if action_porbs is None:
            action_porbs = [0.2, 0.4, 0.2, 0.2]
        choice = random.random()
        print(choice)
        change = 'No change'
        reward = 0
        action_to_take = random.choices(['change node', 'change edge', 'delete edge', 'add edge'],
                                        weights=action_porbs, k=1)[0]
        if action_to_take == 'change node':
            changed_node = int(random.random() * (len(self.current.component_pool) - 3)) + 3
            target_type = int(random.random() * len(self.basic_components))
            self.change_node(changed_node, target_type)
            reward = self.get_reward()
            change = "change node:" + str(changed_node) + " to " + str(target_type)
            return self.current, reward, change
        elif action_to_take == 'change edge':
            e1 = random.choice(list(self.current.graph.keys()))
            if len(self.current.graph[e1]) <= 0:
                return None, -1, None
            e2 = self.current.graph[e1].pop()
            self.current.graph[e1].add(e2)
            reward = self.change_edge(e1, e2)
            change = "change edge:" + str(e1) + " and " + str(e2)
        elif action_to_take == 'delete edge':
            if not self.current.graph_is_valid():
                return None, -1, None
            e1 = random.choice(list(self.current.graph.keys()))
            e2 = self.current.graph[e1].pop()
            self.current.graph[e1].add(e2)
            self.delete_edge([e1, e2])
            reward = self.get_reward()
            change = "delete edge:" + str(e1) + " and " + str(e2)
        elif action_to_take == 'add edge':
            if self.is_terminal():
                return None, -1, None
            e1 = self.current.step - (len(self.current.component_pool) - 3)
            e1 %= len(self.current.port_pool)
            current_edge = e1
            while True:
                self.find_new_edges(e1)
                if len(self.act_vect) == 1 and self.act_vect[0].value == [-1, -1]:
                    e1 += 1
                    e1 %= len(self.current.port_pool)
                    if e1 == current_edge:
                        return None, -1, None
                    continue
                else:
                    action = random.choice(self.act_vect)
                    reward = self.act(action)
                    break
            change = "add edge on port:" + str(e1)
            # TODO random the select process in mutate and print
        return self.current, reward, change


def get_comp_types(sim):
    comp_type_set = []
    for component in sim.current.component_pool:
        for comp_type in sim.basic_components:
            if comp_type in component:
                if comp_type not in comp_type_set:
                    comp_type_set.append(comp_type)
    return comp_type_set


def get_type_components_idx(type_p, sim):
    type_components_idx = []
    for i in range(len(sim.current.component_pool)):
        component = sim.current.component_pool[i]
        if type_p in component:
            type_components_idx.append(i)
    return type_components_idx


def crossover(sim_0, sim_1):
    comp_type_set_0 = get_comp_types(sim_0)
    comp_type_set_1 = get_comp_types(sim_1)
    common_comp_types = list(set(comp_type_set_0).intersection(set(comp_type_set_1)))

    type_p = random.choice(common_comp_types)
    cv_sim_0_comp_idx = random.choice(get_type_components_idx(type_p, sim_0))
    cv_sim_1_comp_idx = random.choice(get_type_components_idx(type_p, sim_1))

    cv_sim_0_port = min(sim_0.current.comp2port_mapping(cv_sim_0_comp_idx)) if random.random() < 0.5 \
        else max(sim_0.current.comp2port_mapping(cv_sim_0_comp_idx))
    cv_sim_1_port = min(sim_1.current.comp2port_mapping(cv_sim_1_comp_idx)) if random.random() < 0.5 \
        else max(sim_1.current.comp2port_mapping(cv_sim_1_comp_idx))

    ports_on_port_0 = list(sim_0.current.graph[cv_sim_0_comp_idx])
    edges_on_port_0 = [[cv_sim_0_port, port] for port in ports_on_port_0]
    ports_on_port_1 = list(sim_1.current.graph[cv_sim_1_comp_idx])
    edges_on_port_1 = [[cv_sim_1_port, port] for port in ports_on_port_1]

    # Crossover
    for edge in edges_on_port_0:
        sim_0.delete_edge(edge)
    for edge in edges_on_port_1:
        sim_0.add_edge(edge)
    for edge in edges_on_port_1:
        sim_1.delete_edge(edge)
    for edge in edges_on_port_0:
        sim_1.add_edge(edge)

    if random.choice([True, False]):
        return sim_0.current, sim_0.get_reward(), "crossover sim_0:" + str(cv_sim_0_port) + " and " + str(cv_sim_1_port)
    else:
        return sim_1.current, sim_1.get_reward(), "crossover sim_1:" + str(cv_sim_0_port) + " and " + str(cv_sim_1_port)

# end Genetic Search related functions


# def get_effi_info_using_anal(self):
#
#     if not self.is_terminal():
#         self.current.parameters = -1
#         self.reward = 0
#         return {'parameter': -1, 'efficiency': 0, 'output_voltage': -500}
#
#     list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
#                                                                                   self.current.component_pool,
#                                                                                   self.current.port_pool,
#                                                                                   self.current.parent,
#                                                                                   self.current.comp2port_mapping)
#     key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
#     if self.graph_2_reward.__contains__(key + '$' + str(self.current.parameters)):
#         efficiency = self.graph_2_reward[key + '$' + str(self.current.parameters)][1]
#         vout = self.graph_2_reward[key + '$' + str(self.current.parameters)][2]
#         effi = {'efficiency': efficiency, 'output_voltage': vout}
#         para = self.graph_2_reward[key + '$' + str(self.current.parameters)][0]
#         effis = {'parameter': para, 'efficiency': effi, 'output_voltage': vout}
#         return effis
#     else:
#         if self.configs_['skip_sim'] and \
#                 (key + '$' + str(self.current.parameters) not in self.key_sim_effi_):
#             tmp_reward = 0
#             effi_info = {'efficiency': 0, 'Vout': 500}
#             tmp_para = self.current.parameters
#             tmp_effi = effi_info['efficiency']
#             tmp_vout = effi_info['Vout']
#             self.graph_2_reward[key + '$' + str(tmp_para)] = [tmp_para, tmp_effi, tmp_vout]
#             self.update_key_sta(key, tmp_reward, True)
#             effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#
#         else:
#             para_result = find_one_analytics_result(key=key, key_expression=self.key_expression,
#                                                     graph=self.current.graph,
#                                                     comp2port_mapping=self.current.comp2port_mapping,
#                                                     port2comp_mapping=self.current.port2comp_mapping,
#                                                     idx_2_port=self.current.idx_2_port,
#                                                     port_2_idx=self.current.port_2_idx,
#                                                     parent=self.current.parent,
#                                                     component_pool=self.current.component_pool,
#                                                     same_device_mapping=self.current.same_device_mapping,
#                                                     port_pool=self.current.port_pool,
#                                                     duty_cycle=self.current.parameters)
#             if para_result:
#                 tmp_reward = 0
#                 tmp_para = -1
#                 if -1 in para_result:
#                     self.graph_2_reward[key + '$' + '-1'] = [-1,
#                                                              self.key_expression[key + '$' + '-1']['Efficiency'],
#                                                              self.key_expression[key + '$' + '-1']['Vout']]
#                     tmp_reward = 0
#                     tmp_para = -1
#                     tmp_effi = self.key_expression[key + '$' + '-1']['Efficiency']
#                     tmp_vout = self.key_expression[key + '$' + '-1']['Vout']
#                     effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#
#                 else:
#                     for k, v in para_result.items():
#                         print(k, v)
#                         effi = {'efficiency': float(v[0]), 'output_voltage': float(v[1])}
#                         if tmp_reward <= calculate_reward(effi, self.configs_['target_vout']):
#                             tmp_reward = calculate_reward(effi, self.configs_['target_vout'])
#                             tmp_para = float(k)
#                             tmp_effi = float(v[0])
#                             tmp_vout = float(v[1])
#                     # We only save the highest rewards' parameter, effi and cout(vout)
#                     self.graph_2_reward[key + '$' + str(tmp_para)] = [tmp_para, tmp_effi, tmp_vout]
#                     effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#                 self.reward = tmp_reward
#                 # self.current.parameters = tmp_para
#
#             else:
#                 tmp_reward = 0
#                 tmp_para = -1
#                 para_result = get_one_analytics_result(key_expression=self.key_expression,
#                                                        graph=self.current.graph,
#                                                        comp2port_mapping=self.current.comp2port_mapping,
#                                                        port2comp_mapping=self.current.port2comp_mapping,
#                                                        idx_2_port=self.current.idx_2_port,
#                                                        port_2_idx=self.current.port_2_idx,
#                                                        parent=self.current.parent,
#                                                        component_pool=self.current.component_pool,
#                                                        same_device_mapping=self.current.same_device_mapping,
#                                                        port_pool=self.current.port_pool,
#                                                        duty_cycle=self.current.parameters,
#                                                        target_vout_min=self.configs_['min_vout'])
#                 if para_result:
#                     if 'None' in para_result:
#                         self.graph_2_reward[key + '$' + '-1'] = [-1,
#                                                                  self.key_expression[key + '$' + '-1']['Efficiency'],
#                                                                  self.key_expression[key + '$' + '-1']['Vout']]
#                         self.current.parameters = -1
#                         tmp_reward = 0
#                         tmp_para = -1
#                         tmp_effi = self.key_expression[key + '$' + '-1']['Efficiency']
#                         tmp_vout = self.key_expression[key + '$' + '-1']['Vout']
#                     else:
#                         for k, v in para_result.items():
#                             effi = {'efficiency': float(v[0]), 'output_voltage': float(v[1])}
#                             if tmp_reward <= calculate_reward(effi, self.configs_['target_vout']):
#                                 tmp_reward = calculate_reward(effi, self.configs_['target_vout'])
#                                 tmp_para = float(k)
#                                 tmp_effi = float(v[0])
#                                 tmp_vout = float(v[1])
#                         self.graph_2_reward[key + '$' + str(tmp_para)] = [tmp_para, tmp_effi, tmp_vout]
#                     effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#                 else:
#                     effis = {'parameter': -1, 'efficiency': 0, 'output_voltage': -500}
#
#     return effis

# def get_effi_info_using_sim(self):
#     if not self.is_terminal():
#         self.current.parameters = -1
#         self.reward = 0
#         return {'parameter': -1, 'efficiency': 0, 'output_voltage': -500}
#
#     list_of_node, list_of_edge, netlist, joint_list = gen_topo.convert_to_netlist(self.current.graph,
#                                                                                   self.current.component_pool,
#                                                                                   self.current.port_pool,
#                                                                                   self.current.parent,
#                                                                                   self.current.comp2port_mapping)
#     key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
#     # to isom, relpece graph_2_reward[key] with ...key and the next line
#     if self.graph_2_reward.__contains__(key + '$' + str(self.current.parameters)):
#         efficiency = self.graph_2_reward[key + '$' + str(self.current.parameters)][1]
#         vout = self.graph_2_reward[key + '$' + str(self.current.parameters)][2]
#         effi = {'efficiency': efficiency, 'output_voltage': vout}
#         tmp_reward = calculate_reward(effi, self.configs_['target_vout'])
#
#         self.current.parameters = self.graph_2_reward[key + '$' + str(self.current.parameters)][0]
#         self.reward = tmp_reward
#         self.hash_counter += 1
#         effis = {'parameter': self.current.parameters, 'efficiency': effi, 'output_voltage': vout}
#         return effis
#     # not in the hash table
#     else:
#         if key + '$' + str(self.current.parameters) in self.key_sim_effi_:
#             tmp_effi = self.key_sim_effi_[key + '$' + str(self.current.parameters)][0]
#             tmp_vout = self.key_sim_effi_[key + '$' + str(self.current.parameters)][1]
#             effi = {'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#             tmp_reward = calculate_reward(effi, self.configs_['target_vout'])
#             tmp_para = self.current.parameters
#             effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#             return effis
#         else:
#             # start_time = datetime.datetime.now()
#             if self.configs_['skip_sim']:
#                 tmp_reward = 0
#                 effi_info = {'efficiency': 0, 'Vout': 500}
#                 tmp_para = self.current.parameters
#                 tmp_effi = effi_info['efficiency']
#                 tmp_vout = effi_info['Vout']
#             else:
#                 tmp_reward, sim_effi, sim_vout, tmp_para = \
#                     get_single_topo_sim_result(current=self.current, sweep=self.configs_['sweep'],
#                                                candidate_params=self.candidate_params,
#                                                key_sim_effi_=self.key_sim_effi_,
#                                                skip_sim=self.configs_['skip_sim'],
#                                                key_expression_mapping=self.key_expression,
#                                                target_vout=self.configs_['target_vout'],
#                                                min_vout=self.configs_['min_vout'])
#                 # print("find_one_simulate_result using: ", str((datetime.datetime.now() - start_time).seconds))
#                 if tmp_para != -1:
#                     tmp_effi = sim_effi
#                     tmp_vout = sim_vout
#                     # self.graph_2_reward[key] = [tmp_para, tmp_effi, tmp_vout]
#                 else:
#                     tmp_para = -1
#                     tmp_reward = 0
#                     tmp_effi = 0
#                     tmp_vout = -500
#                 self.key_sim_effi_[key + '$' + str(tmp_para)] = [tmp_effi, tmp_vout]
#         self.graph_2_reward[key + '$' + str(tmp_para)] = [tmp_para, tmp_effi, tmp_vout]
#         effis = {'parameter': tmp_para, 'efficiency': tmp_effi, 'output_voltage': tmp_vout}
#
#         self.reward = tmp_reward
#         self.current.parameters = tmp_para
#     return effis

# Start the default policy
