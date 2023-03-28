import _thread
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
# from simulator.build_topology import nets_to_ngspice_files
# from simulator.simulation import simulate_topologies
# from simulator.simulation_analysis import analysis_topologies
from simulator3component.build_topology import nets_to_ngspice_files
from simulator3component.simulation import simulate_topologies
from simulator3component.simulation_analysis import analysis_topologies
import numpy as np
import datetime

from multiprocessing import Process, Manager, Pipe
from utils.util import init_position, generate_depth_list, generate_traj_List, mkdir, get_sim_configs, del_all_files
import copy
# import affinity
import cProfile
from multiprocessing import Pool

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
	uct_tree.root_.act_vect_ = []
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


def parallel_UCF_test(depth_list, trajectory, configs, date_str):
	out_file_name = "Results/mutitest" +"-"+ date_str + ".txt"
	figure_folder = "figures/" + date_str + "/"
	mkdir(figure_folder)
	sim_configs = get_sim_configs(configs)
	tree_num = configs["tree_num"]
	thread_num = configs["thread_num"]
	conn_main = []
	conn_sub = []
	for i in range(tree_num):
		conn_0, conn_1 = Pipe()
		conn_main.append(conn_0)
		conn_sub.append(conn_1)
	num_games = configs["game_num"]
	uct_tree_list = []
	start_time = datetime.datetime.now()
	plan_time = 0

	fo = open(out_file_name, "w")
	fo.write("maxdepth,num_runs,avgstep\n")

	avg_step_list = []

	init_nums = 1
	for max_depth in depth_list:
		for num_runs in trajectory:
			print("max depth is", max_depth, ",trajectory is", num_runs, "every thread has ",
			      int(num_runs / thread_num), " trajectories")
			avg_steps = 0
			for j in range(0, init_nums):
				print()
				cumulate_reward_list = []
				fo.write("----------------------------------------------------------------------" + "\n")
				uct_simulators = []
				for i in range(0, int(num_games / init_nums)):
					steps = 0
					avg_cumulate_reward = 0
					cumulate_plan_time = 0
					cumulate_send_time = 0
					cumulate_recv_time = 0
					cumulate_get_action_time = 0
					final_reward = 0
					r = 0
					tree_size = 0
					r = 0
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
					threads = []
					for n in range(thread_num):
						t = Process(target=uct_tree_list[n].set_and_plan, args=(conn_sub[n],))
						threads.append(t)
						t.start()

					while not sim.is_terminal():
						fo.write("**************step "+str(sim.current.step)+"state:*****************\n")
						uct_tree.set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
						plan_start = datetime.datetime.now()

						if steps == 0:
							for n in range(thread_num):
								conn_main[n].send(-1)
								conn_main[n].send(uct_tree.root_)
								conn_main[n].send(uct_tree_list[n].sim_.graph_2_reward)
						else:
							for n in range(thread_num):
								conn_main[n].send(action)
								conn_main[n].send(sim.get_state())
								conn_main[n].send(uct_tree_list[n].sim_.graph_2_reward)
						plan_end_0 = datetime.datetime.now()

						for n in range(thread_num):
							n_tree_size = conn_main[n].recv()
							uct_tree_list[n] = conn_main[n].recv()
							depth = conn_main[n].recv()
							fo.write("----tree size of "+str(n)+": "+str(n_tree_size) + "\n")
							fo.write("----depth of "+str(n)+": "+str(depth) + "\n")
							tree_size += n_tree_size
						plan_end_1 = datetime.datetime.now()
						instance_plan_time = (plan_end_1 - plan_start).seconds
						cumulate_plan_time += instance_plan_time
						cumulate_send_time += (plan_end_0 - plan_start).seconds
						plan_time += (plan_end_1 - plan_start).seconds
						cumulate_recv_time += (plan_end_1 - plan_end_0).seconds

						if configs["act_selection"] == "Pmerge":
							action = get_action_from_planners(uct_tree_list, uct_tree, thread_num)
						elif configs["act_selection"] == "Tmerge":
							action = get_action_from_trees(uct_tree_list, uct_tree, thread_num)
						elif configs["act_selection"] == "Vote":
							action = get_action_from_trees_vote(uct_tree_list, uct_tree, thread_num)
						if configs["output"]:
							print("{}-action:".format(steps), end='')
							action.print()
							print("{}-state:".format(steps), end='')
						plan_end_2 = datetime.datetime.now()
						cumulate_get_action_time += (plan_end_2 - plan_end_1).seconds

						r = sim.act(action)

						if sim.get_state().parent:
							if action.type == 'node':
								act_str = 'adding node {}'.format(sim.basic_components[action.value])
							else:
								if action.value[1] < 0 or action.value[0] < 0:
									act_str = 'skip connecting'
								else:
									act_str = 'connecting {} and {}'.format(sim.current.idx_2_port[action.value[0]],
									                                        sim.current.idx_2_port[action.value[1]])

							sim.get_state().visualize(act_str, figure_folder)

						# fo.write("ports:"+str(sim.current.port_pool) + "\n")
						# fo.write("graph:"+str(sim.current.graph) + "\n")
						# for n in range(configs["tree_num"]):
						# 	uct_tree_list[n].update_root_node(action, sim.get_state())
						for tmp_uct_tree in uct_tree_list:
							# fo.write("tree:"+str(tmp_uct_tree)+"\n")
							sim.graph_2_reward.update(tmp_uct_tree.sim_.graph_2_reward)
							# fo.write("hash table size"+str(len(tmp_uct_tree.sim_.graph_2_reward)) + "\n")
							# fo.write("query time "+str(tmp_uct_tree.sim_.query_counter) + "\n")
							# fo.write("hash time "+str(tmp_uct_tree.sim_.hash_counter) + "\n")
						for tmp_uct_tree in uct_tree_list:
							tmp_uct_tree.sim_.graph_2_reward = sim.graph_2_reward
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
					for t in threads:
						t.terminate()
					threads = []
					avg_steps = avg_steps / configs["game_num"]
					fo.write("Final topology of game "+str(i) + ":\n")
					fo.write("port_pool:"+str(sim.current.port_pool) + "\n")
					fo.write("graph"+str(sim.current.graph) + "\n")
					fo.write("efficiency"+str(effis) + "\n")
					total_query = sim.query_counter
					total_hash_query = sim.hash_counter

					for uct_trees in uct_tree_list:
						total_query += uct_trees.sim_.query_counter
						total_hash_query += uct_trees.sim_.hash_counter

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
	print("figures are saved in:" + str(figure_folder)+"\n")
	print("outputs are saved in:" + out_file_name + "\n")
	del_all_files(str(configs['num_component'])+"component_data_random")
	del_all_files("sim_analysis")

	fo.close()
	return
