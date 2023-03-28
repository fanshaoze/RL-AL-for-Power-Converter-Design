"""
Version:
UCT with DP and prohibit path
5 components
analytical evaluation

Feature:
updated DP with configurable basic weight for not in approved paths
"""
from ucts.TopoPlanner import *
import os
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, read_approve_path, read_joint_component_prob
from utils.eliminate_isomorphism import unblc_comp_set_mapping, get_component_priorities
from SimulatorAnalysis import UCT_data_collection
import sys
from SimulatorAnalysis.simulate_with_topology import *


def read_DP_files(configs):
    target_min_vout = -500
    target_max_vout = 500
    if target_min_vout < configs['target_vout'] < 0:
        approved_path_freq = read_approve_path(0.0, '3comp_buck_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "3comp_buck_boost_sim_node_joint_probs.json")
    elif 0 < configs['target_vout'] < 100:
        approved_path_freq = read_approve_path(0.0, '3comp_buck_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "3comp_buck_sim_node_joint_probs.json")
    elif 100 < configs['target_vout'] < target_max_vout:
        approved_path_freq = read_approve_path(0.0, '3comp_boost_sim_path_freqs.json')
        component_condition_prob = read_joint_component_prob(configs['num_component'] - 3,
                                                             None, "3comp_boost_sim_node_joint_probs.json")
    else:
        return None
    return approved_path_freq, component_condition_prob


def update_max_result(max_record, k, max_para_reward, max_effi, max_vout, max_para):
    max_record.clear()
    max_record['key'] = k
    max_record['reward'] = max_para_reward
    max_record['effi'] = max_effi
    max_record['vout'] = max_vout
    max_record['para'] = max_para
    return max_record


def main(name, try_target_vout, result_folder, trajs, rounds):
    path = './SimulatorAnalysis/database/analytic-expression.json'
    is_exits = os.path.exists(path)
    if not is_exits:
        UCT_data_collection.key_expression_dict()
    simu_output_results = [[] for i in range(rounds + 1)]

    for trajectory in trajs:
        simu_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]
        out_file_folder = 'Results/' + result_folder + '/'
        out_round_folder = out_file_folder + str(trajectory) + '/'

        out_total_result_file = out_round_folder + 'simu_anal_result.txt'
        fo = open(out_total_result_file, "w")
        fo.write("traj:" + str(trajectory) + "\n")
        for test_idx in range(rounds):
            fo.write("round:" + str(test_idx) + "\n")
            configs = {}
            args_file_name = "config.py"
            from config import uct_configs
            configs = uct_configs
            configs["target_vout"] = try_target_vout
            configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
            configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])
            sim_configs = get_sim_configs(configs)
            approved_path_freq, component_condition_prob = read_DP_files(configs)
            key_expression = UCT_data_collection.read_analytics_result()
            preset_component_num = 0

            component_priorities = get_component_priorities()

            _unblc_comp_set_mapping, _ = unblc_comp_set_mapping(['Sa', 'Sb', 'L', 'C'],
                                                                configs['num_component'] - 3 - preset_component_num)

            sim = TopoPlanner.TopoGenSimulator(sim_configs, approved_path_freq, component_condition_prob,
                                               key_expression, _unblc_comp_set_mapping, component_priorities,
                                               configs['num_component'])

            for_simulation_file = 'round' + '_' + str(test_idx)
            topk_json = json.load(open(out_round_folder + for_simulation_file + '.json'))
            max_sim_record = {'key': None, 'reward': -1, 'effi': -500, 'vout': 0, 'para': -1}
            max_anal_record = {'key': None, 'reward': -1, 'effi': -500, 'vout': 0, 'para': -1}
            # [key, reward, effi, vout, para]
            fo.write('idx' + '\t' + '\t' + 'sim_reward' + '\t' + 'sim_effi' + '\t' + 'sim_vout' + '\t' + 'sim_para' +
                     '\t' + 'anal_reward' + '\t' + 'anal_effi' + '\t' + 'anal_vout' + '\t' + 'anal_para' + "\n")
            topk_idx = 0
            for k, v in topk_json.items():
                # [sim_info, anal_reward, anal_effi, anal_vout, anal_para]
                results_tmp = v[0]
                effi_info = simulate_one_analytics_result(results_tmp)
                max_para_reward, max_para_effi_info, max_para = \
                    find_simu_max_reward_para(effi_info, sim.configs_['target_vout'])

                if max_para_reward > max_sim_record['reward']:
                    max_sim_record = update_max_result(max_sim_record, k, max_para_reward,
                                                       max_para_effi_info['efficiency']
                                                       , max_para_effi_info['Vout'], max_para)
                # if v[1] > max_anal_record['reward']:
                #     max_anal_record = update_max_result(max_anal_record, k, v[1], v[2], v[3], v[4])
                # fo.write(
                #     str(topk_idx) + ',sim:\t' + str(max_para_reward) + '\t' + str(max_para_effi_info['efficiency']) +
                #     '\t' + str(max_para_effi_info['Vout']) + '\t' + str(max_para) +
                #     '\t' + ',anal:\t' + str(v[1]) + '\t' + str(v[2]) + '\t' + str(v[3]) + '\t' + str(v[4]) + "\n")
                fo.write(
                    str(topk_idx) + ',sim:\t' + str(max_para_reward) + '\t' + str(max_para_effi_info['efficiency']) +
                    '\t' + str(max_para_effi_info['Vout']) + '\t' + str(max_para) + "\n")

                topk_idx += 1
            # fo.write('max_sim:\t' + str(max_sim_record['reward']) + '\t' + str(max_sim_record['effi']) +
            #          '\t' + str(max_sim_record['vout']) + '\t' + str(max_sim_record['para']) + '\t' +
            #          'max_anal:\t' + str(max_anal_record['reward']) + '\t' + str(max_anal_record['effi']) +
            #          '\t' + str(max_anal_record['vout']) + '\t' + str(max_anal_record['para']) + "\n")
            fo.write('max_sim:\t' + str(max_sim_record['reward']) + '\t' + str(max_sim_record['effi']) +
                     '\t' + str(max_sim_record['vout']) + '\t' + str(max_sim_record['para']) + "\n")
            simu_result = [max_sim_record['effi'], max_sim_record['vout'],
                           max_sim_record['reward'], max_sim_record['para'],
                           ' ']
            simu_results.append(simu_result)
        for i in range(rounds + 1):
            simu_output_results[i].extend(simu_results[i])
        fo.close()
    simu_out_file_name = "Results/DP-" + str(try_target_vout) + "-simu-" + result_folder + "-" + ".csv"
    with open(simu_out_file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(simu_output_results)
    f.close()

    return


if __name__ == '__main__':
    # read_result('mutitest_50-2021-04-17-16-29-41-37526.txt')
    main('PyCharm', 200, '2021-08-01-01-57-38_740458', [10,15], 2)
