import _thread
import copy
import json
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from utils.eliminate_isomorphism import unblc_comp_set_mapping, get_component_priorities
from SimulatorAnalysis import UCT_data_collection
from SimulatorAnalysis.simulate_with_topology import *

import gc


def merge_act_nodes(dest_act_node, act_node):
    dest_act_node.avg_return_ = dest_act_node.avg_return_ * dest_act_node.num_visits_ + \
                                act_node.avg_return_ * act_node.num_visits_
    dest_act_node.num_visits_ += act_node.num_visits_
    dest_act_node.avg_return_ = dest_act_node.avg_return_ / dest_act_node.num_visits_


def get_action_from_trees(uct_tree_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_tree_list[i].node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_tree_list[i].act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_tree_list[i].node_vect_[j])
            else:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_tree_list[i].act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_tree_list[i].node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_planners(uct_planner_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    uct_tree.root_.node_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_planner_list[i].root_.node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_planner_list[i].root_.act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
            else:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_trees_vote(uct_planner_list, uct_tree, tree_num=4):
    action_nodes = []
    counts = {}
    for i in range(tree_num):
        action_nodes.append(uct_planner_list[i].get_action())
    for i in range(len(action_nodes)):
        tmp_count = 0
        if counts.get(action_nodes[i]) is None:
            for j in range(len(action_nodes)):
                if action_nodes[j].equal(action_nodes[i]):
                    tmp_count += 1
            counts[action_nodes[i]] = tmp_count
    for action, tmp_count in counts.items():
        if tmp_count == max(counts.values()):
            selected_action = action
    return selected_action


def read_DP_files(configs):
    target_min_vout = -500
    target_max_vout = 500
    if target_min_vout < configs['target_vout'] < 0:
        approved_path_freq = read_approve_path(0.0,
                                               './UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_boost_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)

    elif 0 < configs['target_vout'] < 100:
        approved_path_freq = read_approve_path(0.0, './UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./UCT_5_UCB_unblc_restruct_DP_v1/3comp_buck_sim_node_joint_probs.json")

        print(approved_path_freq)
        print(component_condition_prob)


    elif 100 < configs['target_vout'] < target_max_vout:
        approved_path_freq = read_approve_path(0.0, './UCT_5_UCB_unblc_restruct_DP_v1/3comp_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None,
                                                             "./UCT_5_UCB_unblc_restruct_DP_v1/3comp_boost_sim_node_joint_probs.json")
        print(approved_path_freq)
        print(component_condition_prob)
    else:
        return None
    return approved_path_freq, component_condition_prob


def write_info_to_file(fo, sim, effis, avg_cumulate_reward, avg_steps, total_query, total_hash_query, start_time,
                       tree_size, configs):
    fo.write("Final topology of game " + ":\n")
    fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
    fo.write(str(sim.current.parameters) + "\n")
    fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
    fo.write("graph:" + str(sim.current.graph) + "\n")
    fo.write("efficiency:" + str(effis) + "\n")
    fo.write("final reward:" + str(avg_cumulate_reward) + "\n")
    fo.write("step:" + str(avg_steps) + "\n")
    fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
    fo.write("hash query time:" + str(total_hash_query) + "\n")
    end_time = datetime.datetime.now()
    fo.write("end at:" + str(end_time) + "\n")
    fo.write("start at:" + str(start_time) + "\n")
    fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
    fo.write("result with parameter:" + str(sim.current.parameters) + "\n")
    fo.write("----------------------------------------------------------------------" + "\n")
    fo.write("configs:" + str(configs) + "\n")
    fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")
    return fo


def print_and_write_child_info(fo, child_idx, _root, child, child_node, child_state):
    print("action ", child_idx, " :", _root.act_vect_[child_idx].type, "on",
          _root.act_vect_[child_idx].value)
    print("action child ", child_idx, " avg_return:", child.avg_return_)
    print("state child ", child_idx, " reward:", child_node.reward_)
    print("state ", child_idx, "ports:", child_state.port_pool)
    print("state child", child_idx, "graph:", child_state.graph)

    fo.write("action " + str(child_idx) + " :" + str(_root.act_vect_[child_idx].type) + "on" +
             str(_root.act_vect_[child_idx].value) + "\n")
    fo.write("action child " + str(child_idx) + " avg_return:" + str(child.avg_return_) + "\n")
    fo.write("action child " + str(child_idx) + " num_visits_:" + str(child.num_visits_) + "\n")
    fo.write(
        "state child " + str(child_idx) + "child node reward:" + str(child_node.reward_) + "\n")
    fo.write("state child " + str(child_idx) + "ports:" + str(child_state.port_pool) + "\n")
    fo.write("state child " + str(child_idx) + "graph:" + str(child_state.graph) + "\n")
    return fo


def get_multi_k_sim_results(sim, uct_simulators, configs, total_query, avg_query_time, anal_results, simu_results):
    effis = []
    max_sim_reward_results = {}
    for k in configs['topk_list']:
        max_sim_reward_results[k] = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}

    sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
    max_result = sim.reward
    if configs['reward_method'] == 'analytics':
        max_topk = copy.deepcopy(uct_simulators[0].topk)
        for k in configs['topk_list']:
            sim.topk = copy.deepcopy(max_topk[-k:])
            # effis: [reward, effi, vout, para]
            effis = sim.get_reward_using_anal()

            if len(sim.topk) == 0:
                max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                         'max_sim_effi': -1, 'max_sim_vout': -500}
            else:
                max_sim_reward_result = get_simulator_tops_sim_info(sim=sim)
            max_sim_reward_results[k] = max_sim_reward_result
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_result = [effis[1], effis[2], max_result, str(sim.current.parameters), total_query, avg_query_time]
                anal_results[k].append(anal_result)
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, avg_query_time])
    elif configs['reward_method'] == 'simulator':
        effis = sim.get_reward_using_sim()
        max_sim_reward_result = {}
        if len(sim.topk) == 0:
            max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                     'max_sim_effi': -1, 'max_sim_vout': -500}
        else:
            max_sim_reward_result = get_simulator_tops_sim_info(sim=sim)
        for k in configs['topk_list']:
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_results[k].append([effis[1], effis[2], max_result, str(sim.current.parameters),
                                        total_query, avg_query_time])
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, avg_query_time])
    else:
        effis = None
    print("effis of topo:", effis)
    return anal_results, simu_results, effis


def trajs_all_in_first_step(total_step, num_runs):
    steps_traj = []
    for i in range(total_step):
        if i == 0:
            steps_traj.append(total_step * num_runs - (total_step - 1))
        else:
            steps_traj.append(1)
    return steps_traj


def copy_simulators_info(sim, uct_simulators):
    sim.graph_2_reward = uct_simulators[0].graph_2_reward
    sim.current_max = uct_simulators[0].current_max
    sim.no_isom_seen_state_list = uct_simulators[0].no_isom_seen_state_list
    sim.key_expression = uct_simulators[0].key_expression
    sim.key_sim_effi_ = uct_simulators[0].key_sim_effi_
    sim.topk = uct_simulators[0].topk
    sim.new_query_time = uct_simulators[0].new_query_time
    sim.new_query_counter = uct_simulators[0].new_query_counter
    if hasattr(sim, 'surrogate_hash_table') and hasattr(uct_simulators[0], 'surrogate_hash_table'):
        sim.surrogate_hash_table = uct_simulators[0].surrogate_hash_table
    return sim


def get_total_querys(sim, uct_simulators):
    total_query = sim.query_counter
    total_hash_query = sim.hash_counter
    for simulator in uct_simulators:
        total_query += simulator.query_counter
        total_hash_query += simulator.hash_counter
    return total_query, total_hash_query


def pre_fix_topo(sim):
    # For fixed commponent type
    init_nodes = []
    init_edges = []
    # init_nodes = [0,1,2,1,2]
    # init_edges = [[0,9],[1,11],[2,3],[3,7],[4,12],[5,12],[6,8],[10,12]]
    print(init_nodes)
    # init_nodes = [1, 0, 1, 3, 0]
    # init_nodes = [0, 0, 3, 3, 1]
    # init_nodes = [0, 0, 1, 2, 3]
    for node in init_nodes:
        action = TopoPlanner.TopoGenAction('node', node)
        sim.act(action)

    # edges = [[0, 7], [1,10], [2,6], [3,9], [4, 8], [5,9], [-1,-1], [-1,-1]]
    # edges = [[0, 6], [1,3], [2,10], [3,7], [4, 11], [5,8], [6,12], [-1,-1], [8,11],[9,11]
    #     , [-1,-1],[-1,-1],[-1,-1]]

    # edges = [[0, 3], [1, 8], [2, 5], [4, 7], [6, 7]]
    # edges = [[0, 8], [1, 3], [2, 4], [4, 6], [5, 7]]
    for edge in init_edges:
        action = TopoPlanner.TopoGenAction('edge', edge)
        sim.act(action)

    # for action_tmp in sim.act_vect:
    #     print(action_tmp.value)
    return init_nodes, init_edges, sim


def collect_data_with_UCT(trajectory, test_number, configs, result_folder, Sim=None, uct_tree_list=None,
                          keep_uct_tree=False):
    global component_condition_prob
    if Sim is None:
        Sim = TopoPlanner.TopoGenSimulator

    sim_configs = get_sim_configs(configs)
    num_runs = trajectory
    avg_steps, avg_cumulate_reward, steps, cumulate_plan_time, r, tree_size, preset_component_num = \
        0, 0, 0, 0, 0, 0, 0
    cumulate_reward_list, uct_simulators, uct_tree_list = [], [], []

    approved_path_freq, component_condition_prob = read_DP_files(configs)
    key_expression = {}
    key_sim_effi = {}

    component_priorities = get_component_priorities()
    _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                        configs['num_component'] - 3 - preset_component_num)

    isom_topo_dict = json.load(open('TopoCount.json'))
    param = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    param['C'] = [10, 20]
    param['L'] = [100, 50]
    count_start_time = time.time()
    k = 500 * 1000
    while k > 0:
        sim_train = Sim(sim_configs, approved_path_freq,
                        component_condition_prob,
                        key_expression, _unblc_comp_set_mapping, component_priorities,
                        key_sim_effi,
                        None, configs['num_component'])
        isom_topo, key = sim_train.generate_random_topology_without_reward()
        if not isom_topo.graph_is_valid():
            continue
        else:
            if key not in isom_topo_dict:
                C_count = isom_topo.count_map['C']
                L_count = isom_topo.count_map['L']

                isom_topo_dict[key] = [1 * len(param['Duty_Cycle']) *
                                       (len(param['C']) ** C_count) *
                                       (len(param['L']) ** L_count), C_count, L_count]
        total_running_time = time.time() - count_start_time
        k -= 1
        if (total_running_time > 3 * 3600) or (k < 0):
            break
        print('--------------------------------------', len(isom_topo_dict))
        del sim_train
    print('count of topo:', len(isom_topo_dict))
    total_count = 0
    for k, v in isom_topo_dict.items():
        total_count += v[0]
    print('count of topo with parameter:', total_count)
    with open('TopoCount.json', 'w') as f:
        json.dump(isom_topo_dict, f)

    return None
