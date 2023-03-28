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

import config
from ucts import uct
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from utils.eliminate_isomorphism import unblc_comp_set_mapping, get_component_priorities
from SimulatorAnalysis import UCT_data_collection
from SimulatorAnalysis.simulate_with_topology import *
from GNN_gendata.GenerateTrainData import update_dataset
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


def get_multi_k_sim_results(sim, uct_simulators, configs, total_query, avg_query_time, avg_query_number,
                            anal_results, simu_results, save_tops):
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
            # effis: [reward, effi, vout, para], get the max's reward
            effis = sim.get_reward_using_anal()

            if len(sim.topk) == 0:
                max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': [],
                                         'max_sim_effi': -1, 'max_sim_vout': -500}
            else:
                max_sim_reward_result = get_simulator_tops_sim_info(sim=sim)
                # [state, anal_reward, anal_para, key, sim_reward, sim_effi, sim_vout, sim_para]
                top_simus = []
                for top in sim.topk:
                    top_simus.append(top[4])
                save_tops.append(top_simus)
            max_sim_reward_results[k] = max_sim_reward_result
            if effis is not None:
                # anal_result [effi, vout, max result, para, total query]
                anal_result = [effis[1], effis[2], max_result, str(sim.current.parameters), total_query,
                               avg_query_time, avg_query_number]
                anal_results[k].append(anal_result)
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, avg_query_time, avg_query_number])

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
                                        total_query, avg_query_time, avg_query_number])
            simu_results[k].append(
                [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                 max_sim_reward_result['max_sim_reward'],
                 max_sim_reward_result['max_sim_para'],
                 total_query, avg_query_time, avg_query_number])
    else:
        effis = None
    print("effis of topo:", effis)

    return anal_results, simu_results, effis, save_tops


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
    sim.new_query_time += uct_simulators[0].new_query_time
    sim.new_query_counter += uct_simulators[0].new_query_counter
    sim.topk = uct_simulators[0].topk
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
    if config.fixed_components:
        init_nodes = [0, 0, 1, 1, 2]
    else:
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


def serial_UCF_test(trajectory, test_number, configs, result_folder, Sim=None, uct_tree_list=None, keep_uct_tree=False):
    global component_condition_prob
    if Sim is None:
        Sim = TopoPlanner.TopoGenSimulator
        inside_sim = True
    else:
        inside_sim = False

    # need to construct the trees if not provided
    initialize_tree = uct_tree_list is None

    out_file_folder = 'Results/' + result_folder + '/'
    mkdir(out_file_folder)
    out_file_name = out_file_folder + str(trajectory) + '-result.txt'
    out_round_folder = 'Results/' + result_folder + '/' + str(trajectory)
    mkdir(out_round_folder)
    figure_folder = "figures/" + result_folder + "/"
    mkdir(figure_folder)

    # out_file_name = "Results/mutitest_" + str(configs['target_vout']) + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    # figure_folder = "figures/" + result_folder + "/"
    # mkdir(figure_folder)

    sim_configs = get_sim_configs(configs)
    start_time = datetime.datetime.now()

    simu_results = {}
    anal_results = {}
    save_tops = []
    for k in configs['topk_list']:
        simu_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'new_query']]
        anal_results[k] = [['efficiency', 'vout', 'reward', 'DC_para', 'query', 'avg_time', 'new_query']]

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []

    for test_idx in range(test_number):
        fo.write("----------------------------------------------------------------------" + "\n")
        num_runs = trajectory
        avg_steps, avg_cumulate_reward, steps, cumulate_plan_time, r, tree_size, preset_component_num = \
            0, 0, 0, 0, 0, 0, 0
        # cumulate_reward_list, uct_simulators, uct_tree_list = [], [], []
        cumulate_reward_list, uct_simulators = [], []
        if initialize_tree:
            uct_tree_list = []

        approved_path_freq, component_condition_prob = read_DP_files(configs)
        key_expression = UCT_data_collection.read_no_sweep_analytics_result() \
            if configs['use_external_expression_hash'] else {}
        key_sim_effi = UCT_data_collection.read_no_sweep_sim_result()

        component_priorities = get_component_priorities()
        # TODO must be careful, if we delete the random adding of sa,sb,
        #  we also need to change the preset comp number
        _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                            configs['num_component'] - 3 - preset_component_num)
        # for k, v in _unblc_comp_set_mapping.items():
        #     print(k, '\t', v)

        # init outer simulator and tree
        sim = Sim(sim_configs, approved_path_freq,
                  component_condition_prob,
                  key_expression, _unblc_comp_set_mapping, component_priorities,
                  key_sim_effi,
                  None, configs['num_component'])

        isom_topo_dict = {}

        uct_tree = uct.UCTPlanner(sim, -1, num_runs, configs["ucb_scalar"], configs["gamma"],
                                  configs["leaf_value"], configs["end_episode_value"],
                                  configs["deterministic"], configs["rave_scalar"], configs["rave_k"],
                                  configs['component_default_policy'], configs['path_default_policy'])

        uct_simulators.clear()
        # uct_tree_list.clear()
        # init inner simulators and trees
        for n in range(configs["tree_num"]):
            uct_simulators.append(Sim(sim_configs, approved_path_freq,
                                      component_condition_prob,
                                      key_expression, _unblc_comp_set_mapping, component_priorities,
                                      key_sim_effi,
                                      None, configs['num_component']))

            if initialize_tree:
                uct_tree_list.append(
                    uct.UCTPlanner(uct_simulators[n], -1, int(num_runs / configs["tree_num"]),
                                   configs["ucb_scalar"], configs["gamma"], configs["leaf_value"],
                                   configs["end_episode_value"], configs["deterministic"],
                                   configs["rave_scalar"], configs["rave_k"], configs['component_default_policy'],
                                   configs['path_default_policy']))
            else:
                # when keeping the tree, we need to assign new simulators to them (the reward function may have changed)
                uct_tree_list[n].sim_ = uct_simulators[n]

        # For fixed commponent type
        init_nodes, init_edges, sim = pre_fix_topo(sim)

        # set roots
        uct_tree.set_root_node(sim.get_state(), sim.get_actions(), sim.get_next_candidate_components(),
                               sim.get_current_candidate_components(), sim.get_weights(), r, sim.is_terminal())

        if initialize_tree:
            for n in range(configs["tree_num"]):
                uct_tree_list[n].set_root_node(sim.get_state(), sim.get_actions(),
                                               sim.get_next_candidate_components(),
                                               sim.get_current_candidate_components(),
                                               sim.get_weights(), r, sim.is_terminal())

        #  configs['num_component'] - 3 - len(init_nodes) for computing how many component to add
        # + 3 + 2 * (configs['num_component'] - 3) for computing how many ports to consider
        # Here, if not sweep, int(not configs['sweep']) is 1, which means we add one step to chose duty cycle
        total_step = configs['num_component'] - 3 - len(init_nodes) + 3 + \
                     2 * (configs['num_component'] - 3) - len(init_edges) + int(not configs['sweep'])

        # True: for using fixed trajectory on every step(False decreasing trajectory)
        if num_runs < 50:
            steps_traj = get_steps_traj(total_step * num_runs, total_step,
                                        int((num_runs / 10) ** 1.8), 2.7, False)
        else:
            steps_traj = get_steps_traj(total_step * 50, total_step,
                                        int((50 / 10) ** 1.8), 2.7, False)
            steps_traj = [int(step * num_runs / 50.0) for step in steps_traj]


        # just for generate the result of 5 component
        # steps_traj = trajs_all_in_first_step(total_step, num_runs)
        if steps_traj[0] == 1:
            steps_traj[0] = 4
        if sim_configs['debug_traj']:
            steps_traj[0] = 20
            for i in range(1, len(steps_traj)):
                steps_traj[i] = 1
        print(steps_traj)
        # traj_idx = len(init_nodes) + len(init_edges)
        traj_idx = 0
        # return
        # for _ in range(1):
        while not sim.is_terminal():
            plan_start = datetime.datetime.now()

            step_traj = steps_traj[traj_idx]
            step_traj = int(step_traj / configs["tree_num"])
            for n in range(configs["tree_num"]):
                print("sim.current.step", sim.current.step)
                tree_size_tmp, tree_tmp, depth = \
                    uct_tree_list[n].plan(step_traj, False)
                tree_size += tree_size_tmp
                fo.write("increased tree size:" + str(tree_size_tmp) + "\n")

            _root = uct_tree_list[0].root_
            for child_idx in range(len(_root.node_vect_)):
                child = _root.node_vect_[child_idx]
                child_node = child.state_vect_[0]
                fo = print_and_write_child_info(fo, child_idx, _root, child, child_node, child_node.state_)

            plan_end_1 = datetime.datetime.now()
            instance_plan_time = (plan_end_1 - plan_start).seconds
            cumulate_plan_time += instance_plan_time

            action = uct_tree_list[0].get_action()

            if configs["output"]:
                print("{}-action:".format(steps), end='')
                action.print()
                fo.write("take the action: type:" + str(action.type) +
                         " value: " + str(action.value) + "\n")
                print("{}-state:".format(steps), end='')

            r = sim.act(action)

            for n in range(configs["tree_num"]):
                # uct_tree_list[n].update_root_node(action, sim.get_state())
                uct_tree_list[n].update_root_node(action, sim.get_state(), keep_uct_tree=keep_uct_tree)
            avg_cumulate_reward += r
            steps += 1
            traj_idx += 1
            # Here means finishing component adding reset traj idx as num_component-3 to guarantee
            # the total number of edge selection is fixed. The graph == {} means graph is {} before
            # edge adding and not {} after the first step of edge adding
            if len(sim.current.component_pool) == sim.num_component_ and sim.current.graph == {}:
                traj_idx = 5

            print("instant reward:", uct_tree.root_.reward_, "cumulate reward: ", avg_cumulate_reward,
                  "planning time:", instance_plan_time, "cumulate planning time:", cumulate_plan_time)
            fo.write("instant reward:" + str(uct_tree.root_.reward_) + "cumulate reward: " + str(avg_cumulate_reward) +
                     "planning time:" + str(instance_plan_time) + "cumulate planning time:" + str(cumulate_plan_time))

        # set back to the original root before returning it
        uct_tree_list[0].root_ = uct_tree_list[0].original_root_

        # get max
        total_query, total_hash_query = get_total_querys(sim, uct_simulators)
        sim = copy_simulators_info(sim, uct_simulators)

        sim.get_state().visualize(
            "result with parameter:" + str(sim.current.parameters) + " ", figure_folder)

        effis = []
        if inside_sim:
            if sim.new_query_counter != 0:
                avg_query_time = sim.new_query_time / sim.new_query_counter
            else:
                avg_query_time = 0
            anal_results, simu_results, effis, save_tops = \
                get_multi_k_sim_results(sim=sim, uct_simulators=uct_simulators, configs=configs,
                                        total_query=total_query, avg_query_time=avg_query_time,
                                        avg_query_number=sim.new_query_counter, save_tops=save_tops,
                                        anal_results=anal_results, simu_results=simu_results)
        print("##################### finish current")
        cumulate_reward_list.append(avg_cumulate_reward)

        avg_steps += steps
        avg_steps = avg_steps / configs["game_num"]
        fo = write_info_to_file(fo, sim, effis, avg_cumulate_reward, avg_steps, total_query, total_hash_query,
                                start_time, tree_size, configs)

        end_time = datetime.datetime.now()
        avg_step_list.append(avg_steps)

        # UCT_data_collection.save_sta_result(uct_simulators[0].key_sta, 'sta_only_epr.json')
        if configs['save_expression']:
            UCT_data_collection.save_no_sweep_analytics_result(uct_simulators[0].key_expression)
        # TODO save simulation rewards
        if configs['save_simu_results']:
            UCT_data_collection.save_no_sweep_sim_result(uct_simulators[0].key_sim_effi_)

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    for result in anal_results:
        fo.write(str(result) + "\n")
    fo.close()

    # save_reward_hash(sim)
    del result

    gc.collect()

    return {'sim': sim,
            'time': (end_time - start_time).seconds,
            'query_num': total_query,
            'state_list': uct_tree_list[0].get_all_states(),
            'uct_tree': uct_tree,
            'uct_tree_list': uct_tree_list
            }, anal_results, simu_results, save_tops

    # if configs['get_traindata']:
    #     for isom_topo_and_key in sim.no_isom_seen_state_list:
    #         isom_topo, key = isom_topo_and_key[0], isom_topo_and_key[1]
    #         if key + '$' + str(isom_topo.parameters) not in sim.key_sim_effi_:
    #             continue
    #         else:
    #             eff = sim.key_sim_effi_[key + '$' + str(isom_topo.parameters)][0]
    #             vout = sim.key_sim_effi_[key + '$' + str(isom_topo.parameters)][1]
    #             if vout == -500:
    #                 continue
    #         isom_topo_dict[key + '$' + str(isom_topo.parameters)] = [isom_topo, eff, vout, 0, 0]
    #     update_dataset(isom_topo_dict)
