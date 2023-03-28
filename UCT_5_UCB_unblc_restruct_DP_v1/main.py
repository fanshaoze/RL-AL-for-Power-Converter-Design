"""
Version:
UCT with DP and prohibit path
5 components
analytical evaluation

Feature:
updated DP with configurable basic weight for not in approved paths
"""
import csv
import datetime
import json
import os

from utils.util import mkdir, generate_depth_list, remove_tmp_files, save_results_to_csv
from Algorithms.SerialUCF import serial_UCF_test
from Algorithms.CollectData import collect_data_with_UCT
from Algorithms.GeneticSearch import genetic_search


def generate_traj_lists(trajectories, test_number):
    traj_lists = []
    for traj in trajectories:
        try_times = test_number
        traj_list = []
        for _ in range(try_times):
            traj_list.append(traj)
        traj_lists.append(traj_list)
    return traj_lists


def main(name='', traj=None, Sim=None, configs=None, uct_tree_list=None, keep_uct_tree=False):
    mkdir("figures")
    mkdir("Results")

    configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
    configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])
    configs['topk_size'] = max(configs['topk_list'])

    traj_lists = generate_traj_lists(configs['trajectories'], configs['test_number'])
    anal_output_results = {}
    simu_output_results = {}
    save_output_tops = {}
    for k in configs['topk_list']:
        anal_output_results[k] = [[] for _ in range(configs['test_number'] + 1)]
        simu_output_results[k] = [[] for _ in range(configs['test_number'] + 1)]
    print(traj_lists)

    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pid_str = str(os.getpid())
    result_folder = date_str + '_' + pid_str

    if traj == None:
        traj_list = configs['trajectories']
        print(traj_list)
    else:
        traj_list = [traj]

    for traj in traj_list:

        if configs["algorithm"] == "UCF" or configs["algorithm"] == "UCT":
            result, anal_results, simu_results, save_tops = serial_UCF_test(trajectory=traj,
                                                                            test_number=configs['test_number'],
                                                                            result_folder=result_folder,
                                                                            configs=configs,
                                                                            Sim=Sim,
                                                                            uct_tree_list=uct_tree_list,
                                                                            keep_uct_tree=keep_uct_tree)
            if Sim is None:  # just for uct with simulator or analytics
                for k in configs['topk_list']:
                    for i in range(configs['test_number'] + 1):
                        anal_output_results[k][i].extend(anal_results[k][i])
                        simu_output_results[k][i].extend(simu_results[k][i])
                save_output_tops[traj] = save_tops
        elif configs["algorithm"] == "Collect":
            result, anal_results, simu_results, save_tops = collect_data_with_UCT(trajectory=traj,
                                                                                  test_number=configs['test_number'],
                                                                                  result_folder=result_folder,
                                                                                  configs=configs,
                                                                                  Sim=Sim,
                                                                                  uct_tree_list=uct_tree_list,
                                                                                  keep_uct_tree=keep_uct_tree)
            if Sim is None:  # just for uct with simulator or analytics
                for k in configs['topk_list']:
                    for i in range(configs['test_number'] + 1):
                        anal_output_results[k][i].extend(anal_results[k][i])
                        simu_output_results[k][i].extend(simu_results[k][i])
                save_output_tops[traj] = save_tops
        else:
            print('alg does not exist')
    for k in configs['topk_list']:
        anal_out_file_name = "Results/BF-" + str(configs["target_vout"]) + '-' + str(
            k) + "-anal-" + date_str + "-" + str(
            os.getpid()) + ".csv"
        save_results_to_csv(anal_out_file_name, anal_output_results[k])
        simu_out_file_name = "Results/BF-" + str(configs["target_vout"]) + '-' + str(
            k) + "-simu-" + date_str + "-" + str(
            os.getpid()) + ".csv"
        save_results_to_csv(simu_out_file_name, simu_output_results[k])

    simu_top_file_name = "Results/BF-" + str(configs["target_vout"]) + "-save-topks-" + date_str + "-" + str(
        os.getpid()) + ".json"
    with open(simu_top_file_name, 'w') as f:
        json.dump(save_output_tops, f)

    remove_tmp_files()

    return result


if __name__ == '__main__':
    main('PyCharm')
