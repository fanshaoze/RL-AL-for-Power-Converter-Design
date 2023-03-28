import _thread
import gc
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner, uct
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
from Viz.uctViz import delAllFiles, TreeGraph
import numpy as np
import datetime

from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from SimulatorAnalysis import UCT_data_collection
from utils.eliminate_isomorphism import get_component_priorities, unblc_comp_set_mapping
import copy
from SimulatorAnalysis.UCT_data_collection import key_expression_dict


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
                if j < len(uct_planner_list[i].root_.node_vect_) and uct_planner_list[i].root_.node_vect_[
                    j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
            else:
                if j < len(uct_planner_list[i].root_.node_vect_) and uct_planner_list[i].root_.node_vect_[
                    j] is not None:
                    uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
    act_node = uct_tree.get_action()
    # act_node = uct_tree.get_action_rave()
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


def brute_force(depth_list, trajectory, test_number, configs, date_str):
    # key_expression = key_expression_dict()
    result_set = []
    out_file_name = "Results/mutitest" + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)
    want_tree_viz = True
    sim_configs = get_sim_configs(configs)

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    init_nums = 1
    results = []
    avg_reward = 0

    approved_path_freq = read_approve_path(0.0, '5comp_buck_path_freqs.json')
    # approved_path_freq = read_approve_path(0.0)

    print(approved_path_freq)
    component_condition_prob = read_joint_component_prob(configs['num_component'] - 3)
    key_expression = UCT_data_collection.read_analytics_result()
    component_priorities = get_component_priorities()
    _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                        configs['num_component'] - 3 - 0)

    for _ in range(test_number):
        Traj = trajectory
        result_sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                                  key_expression, _unblc_comp_set_mapping, component_priorities,
                                                  configs['num_component'])
        max_result = -1
        sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                           key_expression, _unblc_comp_set_mapping, component_priorities,
                                           configs['num_component'])
        init_state = TopoPlanner.TopoGenState(init=True)
        origin_next_candidate_components = copy.deepcopy(sim.next_candidate_components)
        origin_current_candidate_components = copy.deepcopy(sim.current_candidate_components)
        final_counts = {}
        for _ in range(Traj):
            avg_steps = 0
            print()
            avg_cumulate_reward = 0
            fo.write("----------------------------------------------------------------------" + "\n")
            uct_simulators = []
            tree_size = 0
            sim.set_state(origin_next_candidate_components,origin_current_candidate_components,init_state)
            # For fixed commponent type
            init_nodes = []
            # init_nodes = [random.choice([0, 1])]
            # init_nodes = [0, 1, 3, 2, 0]
            # init_nodes = [0, 0, 1, 2, 3]
            # init_nodes = [0, 0, 3, 3, 1]
            # init_nodes = [1, 1, 2, 3, 3]
            # init_nodes = [1,3,0]
            for e in init_nodes:
                action = TopoPlanner.TopoGenAction('node', e)
                sim.act(action)
            edges = []
            # adj = sim.get_adjust_parameter_set()
            # print(sim.get_adjust_parameter_set())
            # {2: {5}, 5: {2}, 1: {8}, 8: {1}, 0: {3}, 3: {0}, 4: {7}, 7: {4, 6}, 6: {7}})
            # edges = [[0, 3], [1, 8], [2, 5], [-1, -1], [4, 6]]
            # edges = [[0, 3], [1, 8], [2, 5], [-1, -1]]
            # edges = []
            for edge in edges:
                action = TopoPlanner.TopoGenAction('edge', edge)
                r = sim.act(action)

            mc_return = 0
            reward_list = []
            discount = 1
            final_return = None
            final_result = sim.default_policy(mc_return, configs["gamma"], discount, reward_list,
                                              configs["component_default_policy"], configs["path_default_policy"])
            if final_result > max_result:
                print(final_result, max_result)
                # result_sim.set_state(copy.deepcopy(sim.get_state()))
                max_result = final_result

            component_types = []
            for component in sim.current.component_pool[3:]:
                if ('GND' not in component) and ('VIN' not in component) and ('VOUT' not in component):
                    component_types.append(uct.get_component_type(component))
            sorted_comp_set = uct.sort_components(tuple(component_types), sim.component_priority)
            if tuple(sorted_comp_set) in final_counts:
                final_counts[tuple(sorted_comp_set)] += 1
            else:
                final_counts[tuple(sorted_comp_set)] = 1
        for k, v in sim.set_count_mapping.items():
            if k in final_counts:
                print(k, '\t', final_counts[k])
            else:
                print(k, '\t', 0)

        # for k, v in final_counts.items():
        #     print(k, '\t', v)
        return

        sim.set_state(result_sim.get_state())
        final_para_str = sim.current.parameters
        sim.get_state().visualize(
            "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)

        effis = [sim.get_effi_info()]
        fo.write("Final topology of game " + ":\n")
        fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
        fo.write(str(sim.current.parameters) + "\n")
        fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
        fo.write("graph:" + str(sim.current.graph) + "\n")
        fo.write("efficiency:" + str(effis) + "\n")
        fo.write("final reward:" + str(max_result) + "\n")
        fo.write("step:" + str(avg_steps) + "\n")
        total_query = sim.query_counter
        total_hash_query = sim.hash_counter
        for simulator in uct_simulators:
            total_query += simulator.query_counter
            total_hash_query += simulator.hash_counter
        fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
        fo.write("hash query time:" + str(total_hash_query) + "\n")
        end_time = datetime.datetime.now()
        final_para_str = sim.current.parameters
        sim.get_state().visualize(
            "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)
        fo.write("end at:" + str(end_time) + "\n")
        # fo.write("start at:" + str(start_time) + "\n")
        # fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
        fo.write("result with parameter:" + str(str(final_para_str)) + "\n")
        fo.write("----------------------------------------------------------------------" + "\n")
        # print(max_depth, ",", num_runs, ":", avg_steps)
        # avg_step_list.append(avg_steps)

        fo.write("configs:" + str(configs) + "\n")
        fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")

        # result = "Traj: " + str(num_runs)
        print(effis)
        result = "Traj: " + str(Traj)
        result = result + "#Efficiency:" + str(effis[0]['efficiency']) + "#FinalRewards:" + str(
            avg_cumulate_reward) + "#QueryTime:" + str(total_query) + "#TreeSize:" + str(tree_size)
        results.append(result)
        UCT_data_collection.save_no_sweep_analytics_result(sim.key_expression)
        UCT_data_collection.save_sta_result(sim.key_sta, 'sta_BF.json')
        # sum = 0
        # for k, v in sim.key_sta.items():
        #     sum += v[0]
        # print('sum', sum)

        del sim
        del result_sim
        gc.collect()

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    for result in results:
        fo.write(result + "\n")
    fo.close()
    return max_result
