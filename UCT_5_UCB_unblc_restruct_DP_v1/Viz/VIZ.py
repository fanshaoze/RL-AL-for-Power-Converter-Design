import os
import random
import ucts.uct
import collections
import networkx as nx
import matplotlib.pyplot as plt
from Viz.uctViz import delAllFiles, TreeGraph
from copy import deepcopy
from ucts import uct
from ucts import TopoPlanner
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
import datetime
from utils.util import init_position, generate_depth_list, del_all_files, mkdir, get_sim_configs


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


def viz_test(depth_list, trajectory, test_number, configs, date_str):
    out_file_name = "Results/mutitest" + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + date_str + "/"
    viz_folder = "Viz/TreeStructures" + date_str + "-" + str(os.getpid()) + "/"
    mkdir(figure_folder)
    mkdir(viz_folder)
    sim_configs = get_sim_configs(configs)
    uct_tree_list = []
    cumulate_reward_list = []
    start_time = datetime.datetime.now()
    viz_step = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # viz_step = [0, 1]
    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []
    init_nums = 1
    for max_depth in depth_list:
        for _ in range(test_number):
            num_runs = trajectory
            print("max depth is", max_depth, ",trajectory is", num_runs, "every thread has ",
                  int(num_runs / configs["tree_num"]), " trajectories")
            avg_steps = 0

            for j in range(0, init_nums):
                print()
                cumulate_reward_list = []
                fo.write("----------------------------------------------------------------------" + "\n")
                uct_simulators = []
                for i in range(0, int(configs["game_num"] / init_nums)):
                    steps = 0
                    avg_cumulate_reward = 0
                    cumulate_plan_time = 0
                    final_reward = 0
                    r = 0
                    tree_size = 0

                    sim = TopoPlanner.TopoGenSimulator(sim_configs, configs['num_component'])
                    uct_tree = uct.UCTPlanner(sim, max_depth, num_runs, configs["ucb_scalar"], configs["gamma"],
                                              configs["leaf_value"], configs["end_episode_value"],
                                              configs["deterministic"])
                    uct_simulators.clear()
                    uct_tree_list.clear()
                    for n in range(configs["tree_num"]):
                        uct_simulators.append(TopoPlanner.TopoGenSimulator(sim_configs, configs['num_component']))
                        uct_tree_list.append(
                            uct.UCTPlanner(uct_simulators[n], max_depth, int(num_runs / configs["tree_num"]),
                                           configs["ucb_scalar"],
                                           configs["gamma"],
                                           configs["leaf_value"], configs["end_episode_value"],
                                           configs["deterministic"]))

                    # For fixed commponent type
                    init_nodes = [0, 3, 1]
                    for e in init_nodes:
                        action = TopoPlanner.TopoGenAction('node', e)
                        sim.act(action)
                    # edges = [[0, 4], [1, 8], [2, 5], [3, 7], [6, 7]]
                    edges = []
                    # for edge in edges:
                    #     action = TopoPlanner.TopoGenAction('edge',edge)
                    #     sim.act(action)
                    sim.get_state().print()
                    uct_tree.set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
                    for n in range(configs["tree_num"]):
                        uct_tree_list[n].set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
                    while not sim.is_terminal():
                        # fo.write(str(steps)+"------step ---------------------------------------------" + "\n")
                        plan_start = datetime.datetime.now()

                        for n in range(configs["tree_num"]):
                            print("sim.current.step", sim.current.step)
                            tree_size_tmp, tree_tmp, depth, node_list = uct_tree_list[n].plan(True, sim.current.step - (
                                        len(edges) + len(init_nodes)))
                            if steps in viz_step:
                                folder = viz_folder + "tree" + str(n) + "step" + str(steps) + "/"
                                is_exists = os.path.exists(folder)
                                if not is_exists:
                                    os.makedirs(folder)
                                else:
                                    delAllFiles(folder)
                                treeviz = TreeGraph(node_list)
                                treeviz.drawAll(uct_tree_list[n], folder, True)
                                tree_size += tree_size_tmp
                                if viz_step.index(steps) == len(viz_step) - 1:
                                    return
                            # fo.write("tree size:"+str(tree_size_tmp)+"\n")
                            # fo.write("hash table size"+str(len(uct_tree_list[n].sim_.graph_2_reward)) + "\n")
                            # fo.write("query time "+str(uct_tree_list[n].sim_.query_counter) + "\n")
                            # fo.write("hash time "+str(uct_tree_list[n].sim_.hash_counter) + "\n")
                            # fo.write("depth "+str(depth) + "\n")

                        print("save roots and child values:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                        fo.write("save roots and child values:&&&&&&&&&&&&&&&&&&&&\n")

                        _root = uct_tree_list[0].root_
                        for child_idx in range(len(_root.node_vect_)):
                            child = _root.node_vect_[child_idx]
                            print("action ", child_idx, " :", _root.act_vect_[child_idx].type,"on",_root.act_vect_[child_idx].value)
                            fo.write("action "+str(child_idx)+" :" + str(_root.act_vect_[child_idx].type) + "on" +
                                     str(_root.act_vect_[child_idx].value)+"\n")
                            print("action child ", child_idx, " avg_return:", child.avg_return_)
                            fo.write("action child " + str(child_idx) + " avg_return:" + str(child.avg_return_)+"\n")
                            child_node = child.state_vect_[0]
                            child_state = child_node.state_
                            print("state child ", child_idx, " reward:", child_node.reward_)
                            fo.write("state child " + str(child_idx) + " reward:" + str(child_node.reward_)+"\n")
                            print("state ", child_idx, "ports:", child_state.port_pool)
                            fo.write("state child " + str(child_idx) + "ports:" + str(child_state.port_pool)+"\n")
                            print("state child", child_idx, "graph:", child_state.graph)
                            fo.write("state child " + str(child_idx) + "graph:" + str(child_state.graph)+"\n")

                        # _root = uct_tree_list[0].root_
                        # print(uct_tree_list[0].root_)
                        # print("end roots and child values:&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                        plan_end_1 = datetime.datetime.now()
                        instance_plan_time = (plan_end_1 - plan_start).seconds
                        cumulate_plan_time += instance_plan_time

                        if configs["act_selection"] == "Pmerge":
                            action = get_action_from_planners(uct_tree_list, uct_tree, configs["tree_num"])
                        elif configs["act_selection"] == "Tmerge":
                            action = get_action_from_trees(uct_tree_list, uct_tree, configs["tree_num"])
                        elif configs["act_selection"] == "Vote":
                            action = get_action_from_trees_vote(uct_tree_list, uct_tree, configs["tree_num"])

                        if configs["output"]:
                            print("{}-action:".format(steps), end='')
                            action.print()
                            print("{}-state:".format(steps), end='')

                        r = sim.act(action)

                        if sim.get_state().parent:
                            if action.type == 'node':
                                act_str = 'adding node {}'.format(sim.basic_components[action.value])
                            elif action.type == 'edge':
                                if action.value[1] < 0 or action.value[0] < 0:
                                    act_str = 'skip connecting'
                                else:
                                    act_str = 'connecting {} and {}'.format(sim.current.idx_2_port[action.value[0]],
                                                                            sim.current.idx_2_port[action.value[1]])
                            else:
                                act_str = 'terminal'

                            sim.get_state().visualize(act_str, figure_folder)
                        for n in range(configs["tree_num"]):
                            uct_tree_list[n].update_root_node(action, sim.get_state())
                        final_reward = r
                        avg_cumulate_reward += r
                        steps += 1
                        print("instant reward:", uct_tree.root_.reward_, "cumulate reward: ", avg_cumulate_reward,
                              "planning time:", instance_plan_time, "cumulate planning time:", cumulate_plan_time)

                    topologies = [sim.get_state()]
                    nets_to_ngspice_files(topologies, configs, configs['num_component'])
                    simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
                    effis = analysis_topologies(configs, len(topologies), configs['num_component'])
                    print("effis of topo:", effis)
                    print("#####################Game:", i, "  steps: ", steps, "  average cumulate reward: ",
                          avg_cumulate_reward)
                    cumulate_reward_list.append(avg_cumulate_reward)
                    avg_steps += steps

                    avg_steps = avg_steps / configs["game_num"]
                    fo.write("Final topology of game " + str(i) + ":\n")
                    fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
                    fo.write("graph" + str(sim.current.graph) + "\n")
                    fo.write("efficiency" + str(effis) + "\n")
                    total_query = sim.query_counter
                    total_hash_query = sim.hash_counter
                    for simulator in uct_simulators:
                        total_query += simulator.query_counter
                        total_hash_query += simulator.hash_counter
                    fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
                    fo.write("hash query time:" + str(total_hash_query) + "\n")
                    end_time = datetime.datetime.now()
                    sim.get_state().visualize(
                        "final of" + str(max_depth) + " " + str(num_runs) + " " + str(i) + " " + out_file_name,
                        figure_folder)
                    fo.write("end at:" + str(end_time) + "\n")
                    fo.write("start at:" + str(start_time) + "\n")
                    fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
                    fo.write("----------------------------------------------------------------------" + "\n")
                    print(max_depth, ",", num_runs, ":", avg_steps)
                    avg_step_list.append(avg_steps)
                fo.write("configs:" + str(configs) + "\n")
                fo.write("final rewards:" + str(final_reward) + "\n")
    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")
    del_all_files(str(configs['num_component']) + "component_data_random")
    del_all_files("sim_analysis")

    fo.close()
    return
