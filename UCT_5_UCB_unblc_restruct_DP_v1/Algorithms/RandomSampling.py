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
from ucts.GetReward import *
import datetime

import numpy as np
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from SimulatorAnalysis import UCT_data_collection

import gc
import datetime
from SimulatorAnalysis.UCT_data_collection import *


def data_dict_to_list(data_json_file):
    data_list = []
    for k, v in data_json_file.items():
        data_list.append([k, v])
    return data_list


def random_sampling(traj, test_number, configs, date_str, target_vout_min=-500, target_vout_max=500):
    path = './SimulatorAnalysis/database/analytic-expression.json'
    out_file_name = "Results/mutitest_" + str(configs['target_vout']) + "-" + date_str + "-" + str(os.getpid()) + ".txt"
    figure_folder = "figures/" + date_str + "/"
    mkdir(figure_folder)

    start_time = datetime.datetime.now()
    good_result_num = 0
    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []
    results = []
    keys = []
    data_json_file = json.load(open("./SimulatorAnalysis/database/data.json"))
    # for k, v in data_json_file.items():
    #     if v['key'] not in keys:
    #         print(v['key'])
    #         keys.append(v['key'])
    # print('number of keys:', len(keys))
    data_list = data_dict_to_list(data_json_file)
    # for i in data_list:
    #     print(i)
    # print(len(data_list))
    # return
    expression_json_file = json.load(open("./SimulatorAnalysis/database/expression.json"))
    print('number of expression:', len(expression_json_file))
    print('number of data:', len(data_list))
    key_json_file = json.load(open("./SimulatorAnalysis/database/key.json"))
    print(len(key_json_file))
    # return
    simu_results = []
    with open("./SimulatorAnalysis/database/analytic.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            simu_results.append(row)

    for _ in range(test_number):
        query_number = 0
        hash_number = 0
        max_reward = -1
        max_result = None
        key_expression = {}
        searched_data = {}
        total_data_len = len(data_list)
        print(total_data_len)
        while query_number < traj:
        # for i in range(0, len(data_list)):
            sample_data = random.choice(data_list)
            # sample_data = data_list[i]
            data_fn = sample_data[0]
            str_list_of_node = str(sample_data[1]['list_of_node'])
            str_list_of_edge = str(sample_data[1]['list_of_edge'])
            str_net_list = str(sample_data[1]['netlist'])
            data_graph = (str_list_of_node, str_list_of_edge, str_net_list)
            if data_graph in searched_data:
                hash_number += 1
                continue
            query_number += 1
            searched_data[data_graph] = query_number

            if sample_data[0] not in expression_json_file:
                expression = "Invalid"
                duty_cycle_para = "None"
                reward = 0
                tmp_para = duty_cycle_para
                tmp_result = {'Expression': expression, 'efficiency': 0, 'output_voltage': target_vout_min}
                if data_json_file[data_fn]['key'] + '$' + duty_cycle_para not in key_expression:
                    key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para] = \
                        {'Expression': expression, 'efficiency': 0, 'output_voltage': target_vout_min}
            else:
                expression = expression_json_file[data_fn]
                for i in range(len(simu_results)):
                    if simu_results[i][0] == data_fn:
                        print(simu_results[i])
                        reward = -1
                        while i < (len(simu_results)) and simu_results[i][0] == data_fn:
                            duty_cycle_para = float(simu_results[i][1])
                            if data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para) not in key_expression:
                                if simu_results[i][3] != 'False' and simu_results[i][4] != 'False':
                                    key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)] = \
                                        {'Expression': expression, 'efficiency': float(simu_results[i][4]) / 100,
                                         'output_voltage': float(simu_results[i][3])}
                                else:
                                    key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)] = \
                                        {'Expression': expression, 'efficiency': 0, 'output_voltage': target_vout_min}

                            effis = key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]
                            tmp_reward = calculate_reward(effis, configs["target_vout"])

                            if tmp_reward > reward:
                                reward = tmp_reward
                                tmp_result = key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]
                                tmp_para = str(duty_cycle_para)
                            i += 1
                        break
                    else:
                        continue
            if tmp_para == '0.5' and tmp_result['efficiency'] > 0.9 and tmp_result['output_voltage'] > 47:
                good_result_num += 1
                print('good_result_num', good_result_num)
                time.sleep(0.5)
            if reward > max_reward:
                max_reward = reward
                result = tmp_result
                para = tmp_para
        print(good_result_num)

        # return

        # topologies = [sim.get_state()]
        effis = [{'efficiency': result['efficiency'], 'output_voltage': result['output_voltage']}]
        print("effis of topo:", effis, " para:", para)
        fo.write("efficiency:" + str(effis) + "\n")
        fo.write("final reward:" + str(max_reward) + "\n")
        fo.write("query time:" + str(query_number) + "\n")
        # sim.get_state().visualize(
        #     "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)
        end_time = datetime.datetime.now()
        fo.write("end at:" + str(end_time) + "\n")
        fo.write("start at:" + str(start_time) + "\n")
        fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
        fo.write("result with parameter:" + str(str(para)) + "\n")
        fo.write("----------------------------------------------------------------------" + "\n")
        fo.write("configs:" + str(configs) + "\n")

        result = "Traj: " + str(traj)
        print(effis, ", ", para)
        if para == '0.5' and effis[0]['efficiency'] > 0.9 and effis[0]['output_voltage'] > 47:
            good_result_num += 1
        result = result + "#efficiency:" + str(effis[0]['efficiency']) + "#vout:" + str(effis[0]['output_voltage']) \
                 + "#para:" + str(para) + "#FinalRewards:" + str(
            max_reward) + "#ExecuteTime:" + str((end_time - start_time).seconds) + "#QueryTime:" + str(
            query_number)
        results.append(result)
        del key_expression
        gc.collect()

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")
    for result in results:
        fo.write(result + "\n")

    fo.write('good prob:' + str(good_result_num / len(traj_list)) + "\n")
    print(good_result_num / len(traj_list))
    fo.close()

    # save_reward_hash(sim)
    del result
    gc.collect()
    return


def read_result(file_name):
    # Traj: 2000#efficiency:0.98#vout:49.0#para:0.5#FinalRewards:0.9023840000000001#ExecuteTime:3321#QueryTime:2000
    file_name = 'Results/'+file_name
    fo_conf = open(file_name, "r")
    line = fo_conf.readline()
    good_count = 0
    total_count = 0
    effis = []
    while True:
        line = fo_conf.readline()
        # print(line)
        if not line:
            break
        if 'Traj: ' in line:
            print(line)
            total_count+=1
            if total_count>100:
                break
            items = line.split('#')
            for item in items:
                if 'efficiency' in item:
                    effi = float(item.split(':')[1])
                elif 'para' in item:
                    para = float(item.split(':')[1])
                elif 'vout' in item:
                    vout = float(item.split(':')[1])
                elif 'Traj' in item:
                    traj = float(item.split(':')[1])
            if effi >= 0.9 and 45.0 < vout < 55:

                good_count += 1
            if effi>40:
                effi = 0.98
            if effi >= 0.9 and (vout < 40.0 or vout > 60):
                effi = 0.01
            effis.append(effi)
    print(effis)
    print(len(effis))
    print("avg:", np.mean(effis))
    print("std error:", np.var(effis)/math.sqrt(total_count))
    print(good_count / 100)


