from copy import deepcopy
import datetime
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from SimulatorAnalysis import UCT_data_collection

import gc

from utils.util import get_sim_configs, mkdir, del_all_files


def read_DP_files(configs):
    target_min_vout = -500
    target_max_vout = 500
    if target_min_vout < configs['target_vout'] < 0:
        approved_path_freq = read_approve_path(0.0, '5comp_buck_coost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "5comp_buck_coost_sim_path_freqs.json")
    elif 0 < configs['target_vout'] < 100:
        approved_path_freq = read_approve_path(0.0, '5comp_buck_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "5comp_buck_sim_path_freqs.json")
    elif 100 < configs['target_vout'] < target_max_vout:
        approved_path_freq = read_approve_path(0.0, '5comp_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "5comp_boost_sim_path_freqs.json")
    else:
        return None
    return approved_path_freq, component_condition_prob


def genetic_search(configs, date_str):
    out_file_name = "Results/mutitest" + "-" + date_str + ".txt"
    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)
    fo = open(out_file_name + "GS", "w")

    num_games = configs["game_num"]
    num_component = configs["num_component"]
    start_time = datetime.datetime.now()
    approved_path_freq, component_condition_prob = read_DP_files(configs)

    sim_configs = get_sim_configs(configs)
    for _ in range(0, num_games):
        sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                           key_expression, configs['num_component'])

        sim.random_generate_graph()
        print("num_component:", sim.current.num_component)
        print("component_pool:", sim.current.component_pool)
        print("port_pool:", sim.current.port_pool)
        print("port2comp_mapping:", sim.current.port2comp_mapping)
        print("comp2port_mapping:", sim.current.comp2port_mapping)
        print("same_device_mapping:", sim.current.same_device_mapping)
        print("idx_2_port:", sim.current.idx_2_port)
        print("port_2_idx:", sim.current.port_2_idx)
        print("graph:", sim.current.graph)
        print("parent:", sim.current.parent)
        print("count_map:", sim.current.count_map)
        print("step:", sim.current.step)
        sim.get_state().visualize("GS init" + out_file_name + "GS", figure_folder)

        return
        print("--------------------------------init state is valid-----------------------------")
        fo.write("-------------------------------init state is valid----------\n")

        topologies = [sim.get_state()]
        nets_to_ngspice_files(topologies, configs, num_component)
        simulate_topologies(len(topologies), num_component, sys_os)
        effis = analysis_topologies(configs, len(topologies), num_component)
        fo.write("init topology:" + "\n")
        print("effis of topo:", effis)
        fo.write(str(sim.current.port_pool) + "\n")
        fo.write(str(sim.current.graph) + "\n")
        fo.write(str(effis) + "\n")
        print("port pool:", str(sim.current.port_pool) + "\n")
        print("graph:", str(sim.current.graph) + "\n")
        print("effi:", str(effis) + "\n")
        init_state = sim.get_state()
        origin_state = deepcopy(init_state)
        origin_query = sim.query_counter
        sim.get_state().visualize("GS init" + out_file_name + "GS", figure_folder)

        mutate_num = configs["mutate_num"]
        generation_num = configs["mutate_generation"]
        individual_num = configs["individual_num"]
        select_num = individual_num / mutate_num
        uct_simulators = []
        individuals = []
        top_topos = []

        def keyFunc(element):
            return element[1]


        init_state = TopoPlanner.TopoGenState(init=True)
        for _ in range(individual_num):
            sim.set_state(deepcopy(init_state))
            final_result = sim.default_policy(mc_return, configs["gamma"], discount, reward_list,
                                              False, False)
            individuals.append([sim.get_state(), final_result])
        individuals.sort(key=keyFunc)  # actually we can use Select(A, n)
        top_topos = individuals[-4:]

        fo.write("generation state with" + str(generation_num) + "\n")
        max_reward = sim.get_reward()
        for j in range(generation_num):
            individuals.clear()
            for top_topo in top_topos:
                i = 0
                individuals.append(top_topo)
                while i < mutate_num:
                    sim.set_state(top_topo[0])
                    mutation_probs = [0.2, 0.4, 0.2, 0.2]
                    mutated_state, reward, change = sim.mutate(mutation_probs)
                    if mutated_state is None and reward == -1:
                        continue
                    if not mutated_state.graph_is_valid():
                        print("not valid graph:", mutated_state.graph)
                    else:
                        individuals.append([mutated_state, reward])
                    fo.write(str(i) + " " + str(j) + " " + change + " reward " + str(reward) + "\n")
                    i += 1
            sim.set_state(deepcopy(init_state))
            final_result = sim.default_policy(mc_return, configs["gamma"], discount, reward_list,
                                              False, False)
            fo.write("random" + " " + str(j) + " " + change + " reward " + str(final_result) + "\n")
            individuals.append([sim.get_state(), final_result])
            individuals.sort(key=keyFunc)  # actually we can use Select(A, n)
            top_topos = individuals[-4:]
            step_best = top_topos[-1]

            fo.write(str(j) + " step GS best: -----------------------------------------" + "\n")
            fo.write(str(step_best[0].port_pool) + "\n")
            fo.write(str(step_best[0].graph) + "\n")
            fo.write("max reward of " + str(j) + " " + str(step_best[1]) + "\n")
        sim.current.visualize("GS result of:" + str(k) + " in " + out_file_name, figure_folder)
        results.append((k, sim.current.port_pool, str(sim.current.graph), str(effis), str(max_reward)))
        fo.write("finish GS of " + str(k) + "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + "\n")
        fo.write("query of " + str(k) + " = " + str(sim.query_counter) + "\n")
        fo.write("hash of " + str(k) + " = " + str(sim.hash_counter) + "\n")
        print("Finished genetic search")
        sim.set_state(origin_state)
        fo.write("--------------------Finished genetic search-------------------\n")
        for k in range(len(results)):
            fo.write("result of " + str(k) + "+++++++++++++++++++++++++++++++" + "\n")
            fo.write(str(results[k][0]) + "\n")
            fo.write(str(results[k][1]) + "\n")
            fo.write(str(results[k][2]) + "\n")
            fo.write(str(results[k][3]) + "\n")
            fo.write(str(results[k][4]) + "\n")
        end_time = datetime.datetime.now()
        fo.write("origin query:" + str(origin_query) + "\n")
        fo.write("end at:" + str(end_time) + "\n")
        fo.write("start at:" + str(start_time) + "\n")
        fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")
    del_all_files(str(configs['num_component']) + "component_data_random")
    del_all_files("sim_analysis")
    fo.close()
    return
