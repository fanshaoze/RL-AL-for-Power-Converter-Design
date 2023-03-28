import os
import sys

sys.path.append(os.path.join(sys.path[0], 'topo_data_util'))
sys.path.append(os.path.join(sys.path[0], 'GNN_gendata'))
sys.path.append(os.path.join(sys.path[0], 'transformer_SVGP'))
sys.path.append(os.path.join(sys.path[0], 'UCT_5_UCB_unblc_restruct_DP_v1'))
import collections
import copy
import csv
import gc
import json
import logging
import random

import time
import math
from datetime import datetime

from transformer_SVGP.GetReward import calculate_reward
from transformer_SVGP.dataset import Dataset
from transformer_SVGP.build_vocab import Vocabulary
import transformer_SVGP.transformer_config
# from transformer_SVGP.transformer_utils import evaluate_model
from matplotlib import pyplot as plt

import numpy as np
# import torch
import config

print(sys.path)
from UCT_5_UCB_unblc_restruct_DP_v1.main import main as run_uct
from UCT_5_UCB_unblc_restruct_DP_v1.ucts.TopoPlanner import get_observed_topology_count

dir = os.path.dirname(__file__)

from uctHelper import recompute_tree_rewards
from al_arguments import get_args
from queryUpdate import query_update
from al_util import feed_random_seeds


# def init_trained_data_keys(dataset):
#     return [key_circuit_from_lists(node_list=dataum['list_of_node'], edge_list=dataum['list_of_edge'],
#                                    net_list=dataum['netlist']) + str(dataum['duty']) for dataum in dataset]


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exits = os.path.exists(path)
    if not is_exits:
        os.makedirs(path)
        print(path + ' created')
        return True
    else:
        print(path + ' already existed')
        return False


def get_topk_rewards(collected_surrogate_rewards, cand_states, k_list, sim):
    """

    @param collected_surrogate_rewards: the surrogate model rewards for hte cnadidate states
    @param cand_states: candidate states
    @param k_list: list of k
    @param sim: simulator, to get the true rewards
    @return: list of rewards for the different k
    """
    top_1_for_klist = {}
    for k in k_list:
        candidate_indices = collected_surrogate_rewards.argsort()[-k:]

        # the true (reward, eff, vout) of the top k topologies decided by the surrogate model
        true_reward_top_k = [sim.get_true_reward(cand_states[idx]) for idx in candidate_indices]
        if not true_reward_top_k:
            print('empty top k')
        # top_1 = max(true_reward_top_k)
        top_1_for_klist[k] = {'reward': max(true_reward_top_k), 'query': len(true_reward_top_k)}
    return top_1_for_klist


def get_topk_rewards_anal(collected_surrogate_rewards, cand_states, k_list, sim,
                          use_pre_simulated=True, target_vout=50):
    """

    @param use_pre_simulated: use the pre simulated sim results
    @param target_vout:
    @param collected_surrogate_rewards: the surrogate model rewards for hte cnadidate states
    @param cand_states: candidate states
    @param k_list: list of k
    @param sim: simulator, to get the true rewards
    @return: list of rewards for the different k
    """
    top_1_for_klist = {}
    for k in k_list:
        candidate_indices = collected_surrogate_rewards.argsort()
        true_reward_top_k = []
        # the true (reward, eff, vout) of the top k topologies decided by the surrogate model
        t, tmp_k = 1, len(cand_states) if k > len(cand_states) else k
        if sim.configs_['sweep']:
            seen_key = []
        while tmp_k > 0:
            cand_state = cand_states[candidate_indices[-t]]
            print(collected_surrogate_rewards[candidate_indices[-t]])
            circuit_key = cand_state.get_key()
            if sim.configs_['sweep']:
                if circuit_key in seen_key:
                    t += 1  # do not decrease tmp k, for sweep the other duty cycle
                    continue
                else:
                    seen_key.append(circuit_key)
            tmp_k = tmp_k - 1
            sim.set_state(None, None, cand_state)
            true_reward_top_k.append(sim.get_real_performance()[0])
            print(len(true_reward_top_k))
            t += 1
        top_1_for_klist[k] = {'reward': max(true_reward_top_k), 'query': len(true_reward_top_k)}
    return top_1_for_klist


def get_topk_rewards_surrogate_model(collected_surrogate_rewards, cand_states, k_list, sim,
                                     count_pre_simulated=False, target_vout=50):
    """

    @param count_pre_simulated: the pre simulated sim results also count in the tops
    @param target_vout:
    @param collected_surrogate_rewards: the surrogate model rewards for hte cnadidate states
    @param cand_states: candidate states
    @param k_list: list of k
    @param sim: simulator, to get the true rewards
    @return: list of rewards for the different k
    """
    top_1_for_klist = {}
    tmp_graph_2_reward = copy.deepcopy(sim.graph_2_reward)
    updated_graph_2_reward = copy.deepcopy(sim.graph_2_reward)
    for k in k_list:
        sim.graph_2_reward = {}
        for tmp_k, tmp_v in tmp_graph_2_reward.items():
            sim.graph_2_reward[tmp_k] = tmp_v
        candidate_indices = collected_surrogate_rewards.argsort()
        true_reward_top_k = []
        # the true (reward, eff, vout) of the top k topologies decided by the surrogate model
        t, tmp_k = 1, len(cand_states) if k > len(cand_states) else k
        if sim.configs_['sweep']:
            seen_key = []
        seen_state_key = []
        while tmp_k > 0:
            if t > len(cand_states): break
            cand_state = cand_states[candidate_indices[-t]]
            print(collected_surrogate_rewards[candidate_indices[-t]])
            circuit_key = cand_state.get_key()
            if sim.configs_['sweep']:
                if circuit_key in seen_key:
                    t += 1  # do not decrease tmp k, for sweep the other duty cycle
                    continue
                else:
                    seen_key.append(circuit_key)
            state_key = circuit_key + '$' + str(cand_state.parameters)
            if state_key not in seen_state_key: seen_state_key.append(state_key)
            # not for anal
            if state_key in sim.graph_2_reward:
                true_reward_top_k.append(
                    calculate_reward({'efficiency': float(sim.graph_2_reward[state_key][1]),
                                      'output_voltage': float(sim.graph_2_reward[state_key][2])},
                                     target_vout))
                if count_pre_simulated:
                    tmp_k = tmp_k - 1
            else:
                tmp_k = tmp_k - 1
                true_reward_top_k.append(sim.get_true_reward(cand_state))
                print(len(true_reward_top_k))
            t += 1
        for circuit_key, v in sim.graph_2_reward.items():
            if circuit_key not in seen_state_key:
                true_reward_top_k.append(calculate_reward({'efficiency': float(v[1]), 'output_voltage': float(v[2])},
                                                          target_vout))
        updated_graph_2_reward.update(sim.graph_2_reward)

        top_1_for_klist[k] = {'reward': max(true_reward_top_k), 'query': len(true_reward_top_k)}
        tmp_k = len(cand_states) if k > len(cand_states) else k
    sim.graph_2_reward = updated_graph_2_reward
    return top_1_for_klist


def model_random_sampling_statistic(sim):
    """
    Save the raw data to the file to plot the ground_truth-prediction graph
    @param file_name: file name of the raw data, [model_name] + '-' + str(number of data) +
    '-' + '[ground truth name]_as_gt'
    @param raw_data: raw_data = {'pred_rewards': pred_rewards, 'ground_truth': ground_truth},
    it depends on which you want to treat as ground truth
    @return: None
    """
    dataset = json.load(open('dataset_5_05_fix_comp.json'))
    raw_data = {'pred_rewards': [], 'ground_truths': []}
    top_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for k, v in dataset.items():
        _, _, pred_reward = sim.get_surrogate_reward_with_topo_info(edge_list=v["list_of_edge"],
                                                                    node_list=v["list_of_node"],
                                                                    duty=v['duty_cycle'])
        raw_data['pred_rewards'].append(pred_reward)

        eff_obj = {'efficiency': v['eff'], 'output_voltage': v['vout']}
        ground_truth = calculate_reward(eff_obj, sim.configs_['target_vout'])
        raw_data['ground_truths'].append(ground_truth)
    sample_length = 1000
    results = {}
    for top_ratio in top_ratios:
        results[top_ratio] = []
    for _ in range(200):
        predict_rewards = random.sample(raw_data['pred_rewards'], sample_length)
        pred_reward_for_sort = np.array(predict_rewards)
        for top_ratio in top_ratios:
            candidate_indices = pred_reward_for_sort.argsort()[-int(top_ratio * sample_length):]

            # the true (reward, eff, vout) of the top k topologies decided by the surrogate model
            true_reward_top_k = [raw_data['ground_truths'][idx] for idx in candidate_indices]
            good_sum = 0
            for true_reward in true_reward_top_k:
                if true_reward > 0.5:
                    good_sum += 1
            results[top_ratio].append(good_sum)
    results_mean = []
    for top_ratio in top_ratios:
        print(top_ratio, sum(results[top_ratio]) / 200)
        results_mean.append(sum(results[top_ratio]) / 200)
        print(results[top_ratio])
    return results, results_mean

    # print(len(raw_data['ground_truths']))


def distribution_plot(evaluation, prediction, file_name):
    """
    plot the distribution
    @param evaluation:
    @param prediction:
    @param file_name:
    @return:
    """
    # Use a breakpoint in the code line below to debug your script.
    x = evaluation
    y = evaluation
    y_1 = prediction
    plt.title("")

    plt.xlabel("evaluation")
    plt.ylabel("prediction")
    plt.plot(x, y, 'ob', color='b', markersize='2')
    plt.plot(x, y_1, 'ob', color='y', markersize='2')

    plt.savefig(file_name)


def save_raw_data_to_files(file_name, sim):
    """
    Save the raw data to the file to plot the ground_truth-prediction graph
    @param file_name: file name of the raw data, [model_name] + '-' + str(number of data) +
    '-' + '[ground truth name]_as_gt'
    @param raw_data: raw_data = {'pred_rewards': pred_rewards, 'ground_truth': ground_truth},
    it depends on which you want to treat as ground truth
    @return: None
    """
    if config.fixed_components:
        dataset = json.load(open('dataset_5_05_fix_comp.json'))
    else:
        dataset = json.load(open('dataset_5_05.json'))
    dataset = json.load(open('dataset_5_fix_comp.json'))
    raw_data = {'pred_rewards': [], 'ground_truths': []}
    for k, v in dataset.items():
        _, _, pred_reward = sim.get_surrogate_reward_with_topo_info(edge_list=v["list_of_edge"],
                                                                    node_list=v["list_of_node"],
                                                                    duty=v['duty_cycle'])
        raw_data['pred_rewards'].append(pred_reward)

        eff_obj = {'efficiency': v['eff'], 'output_voltage': v['vout']}
        ground_truth = calculate_reward(eff_obj, sim.configs_['target_vout'])
        raw_data['ground_truths'].append(ground_truth)
        # print(len(raw_data['ground_truths']))

    raw_data_csv = []
    for i in range(len(raw_data['pred_rewards'])):
        raw_data_csv.append([raw_data['pred_rewards'][i], raw_data['ground_truths'][i]])
    header = ['pred_rewards', 'ground_truths']
    with open(file_name + '.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(raw_data_csv)
    distribution_plot(evaluation=raw_data['ground_truths'], prediction=raw_data['pred_rewards'], file_name=file_name)
    f.close()


def save_prediction_raw_data(sim, iteration, avg_distribution_raw_data):
    """
    Save the raw data to the file to plot the ground_truth-prediction graph
    @param avg_distribution_raw_data:
    @param sim:
    @param iteration:
    it depends on which you want to treat as ground truth
    @return: None
    """
    if config.fixed_components:
        dataset = json.load(open('dataset_5_05_fix_comp.json'))
    else:
        dataset = json.load(open('dataset_5_05.json'))
    dataset = json.load(open('dataset_fix_comp.json'))
    raw_data = {'pred_rewards': [], 'ground_truths': []}
    t = 0
    for k, v in dataset.items():
        _, _, pred_reward = sim.get_surrogate_reward_with_topo_info(edge_list=v["list_of_edge"],
                                                                    node_list=v["list_of_node"],
                                                                    duty=v['duty_cycle'])
        raw_data['pred_rewards'].append(pred_reward)

        eff_obj = {'efficiency': v['eff'], 'output_voltage': v['vout']}
        # ground_truth = calculate_reward(eff_obj, sim.configs_['target_vout'])
        # raw_data['ground_truths'].append(ground_truth)
        # print(len(raw_data['ground_truths']))
        t += 1
        if t % 500 == 0:
            print(t)

    avg_distribution_raw_data[iteration].append(raw_data['pred_rewards'])
    return avg_distribution_raw_data


def save_average_raw_data_to_file(avg_distribution_raw_data, sim, file_name):
    """

    @param avg_distribution_raw_data:
    @param sim:
    @param file_name:
    @return:
    """
    ground_truth_data, raw_data = [], {}
    dataset = json.load(open('dataset_5_05_fix_comp.json'))
    for k, v in dataset.items():
        eff_obj = {'efficiency': v['eff'], 'output_voltage': v['vout']}
        ground_truth = calculate_reward(eff_obj, sim.configs_['target_vout'])
        ground_truth_data.append(ground_truth)

    for iteration in avg_distribution_raw_data:
        raw_data_csv = []
        c = np.array(avg_distribution_raw_data[iteration])
        raw_data['pred_rewards'] = c.mean(axis=0)
        for i in range(len(raw_data['pred_rewards'])):
            raw_data_csv.append([raw_data['pred_rewards'][i], ground_truth_data[i]])
        header = ['pred_rewards', 'ground_truths']
        with open(file_name + '_iter_' + str(iteration) + '.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            csv_writer.writerows(raw_data_csv)
        distribution_plot(evaluation=ground_truth_data, prediction=raw_data['pred_rewards'],
                          file_name=file_name)
        f.close()


def save_results_to_csv(traj_rows, result_folder, output_name):
    for k in traj_rows:
        with open(result_folder + output_name + '-' + str(k) + '.csv', 'w') as f:
            csv_writer = csv.writer(f)
            header = []
            for _ in range(len(args.traj_list)):
                header.extend(['seed', 'query num', 'reward', 'time', 'traj',
                               'retrain_duration', 'top_query', 'cumulate query',
                               'retrain_query', 'simu_if_no_hash', 'real_simulation', 'ensemble batch time'])
            csv_writer.writerow(header)
            for iter_num in traj_rows[k].keys():
                csv_writer.writerow(np.mean(traj_rows[k][iter_num], axis=0))


# def model_topk_evaluation(factory, device, seed, iter_num, output_file):
#     """
#
#     @param output_file: topk save in the file name
#     @param factory: the factory that has the model
#     @param device: device
#     @param seed: seed of current testing set
#     @param iter_num: current iteration
#     @return: None, save in the file
#     """
#     top_k_results = evaluate_model(
#         eff_model=dict(model=factory.eff_models.model, gp=factory.eff_models.gp),
#         vout_model=dict(model=factory.vout_models.model, gp=factory.vout_models.gp),
#         vocab=factory.vocab,
#         data='transformer_SVGP/dataset_5_05_fix_comp_all.json',
#         device=device)
#     model_info = 'seed_' + str(seed) + '_iter_' + str(iter_num)
#
#     with open(output_file, 'w') as f:
#         csv_writer = csv.writer(f)
#         csv_writer.writerow([model_info] + list(top_k_results.values()))


def update_configs_with_args(_configs, args):
    _configs['test_number'] = 1

    _configs['skip_sim'] = args.skip_sim
    _configs['sweep'] = args.sweep
    _configs['topk_list'] = args.k_list
    _configs['get_traindata'] = False
    _configs['round'] = args.round
    _configs['gnn_nodes'] = args.gnn_nodes
    _configs['predictor_nodes'] = args.predictor_nodes
    _configs['gnn_layers'] = args.gnn_layers
    _configs['model_index'] = args.model_index
    # _configs['reward_model'] = args.reward_model
    if args.model == 'simulator' or args.model == 'analytics':
        _configs['reward_method'] = args.model
    else:
        _configs['reward_method'] = 'gnn'
    _configs['nnode'] = args.nnode
    _configs['sweep'] = args.sweep
    _configs['algorithm'] = args.algorithm
    _configs['use_external_expression_hash'] = args.use_external_expression_hash
    _configs['using_exp_inner_hash'] = args.using_exp_inner_hash
    _configs['save_expression'] = args.save_expression
    _configs['save_simu_results'] = args.save_simu_results
    _configs['debug_traj'] = args.debug_traj
    _configs['component_default_policy'] = args.component_default_policy
    _configs['path_default_policy'] = args.path_default_policy
    return _configs


def init_output_rows(k_list, traj_list, seed_range):
    """

    @param k_list:
    @param traj_list:
    @param seed_range:
    @return:
    """
    traj_rows = collections.defaultdict(dict)
    for k in k_list:
        traj_rows[k] = collections.defaultdict(list)
    for k in k_list:
        # we assume that all the traj are of the same length, the * is the pre-update rewards and 1* is for post-update
        for iter_num in range(len(traj_list[0])):
            traj_rows[k][iter_num] = [[] for _ in range(len(seed_range))]
    return traj_rows


def init_training_data(_args):
    vocab = Vocabulary()
    vocab.load(_args.vocab)
    return Dataset(data_file_name=_args.training_data_file, vocab=vocab,
                   max_seq_len=transformer_SVGP.transformer_config.max_path_num,
                   label_len=transformer_SVGP.transformer_config.max_path_len)


def sample_small_data(file_name='./transformer_SVGP/data/dataset_5_cleaned_label_dev_0.6_0_small',
                      ratio=0.0055):
    dev_data = json.load(open(file_name + '.json'))
    print(len(dev_data))
    sample_len = int(len(dev_data) * ratio)
    sample_list = random.sample([i for i in range(len(dev_data))], sample_len)
    print(sample_list)
    sample_dev_data = []
    for sample_ind in sample_list:
        sample_dev_data.append(dev_data[sample_ind])
    out_file = open(file_name + '_small.json', "w")
    json.dump(sample_dev_data, out_file)
    exit()


def get_top_for_klist(_args, collected_surrogate_rewards, cand_states, sim, _configs, start_time):
    """
    get the tops of k_list in args
    @param _args:
    @param collected_surrogate_rewards:
    @param cand_states:
    @param sim:
    @param _configs:
    @return:
    """
    tmp_graph_to_reward = copy.deepcopy(sim.graph_2_reward)
    if _args.model == 'transformer' or _args.model == 'gnn':
        top_1_for_klist = get_topk_rewards_surrogate_model(
            collected_surrogate_rewards=np.array(collected_surrogate_rewards),
            cand_states=cand_states, k_list=args.k_list, sim=sim, count_pre_simulated=False,
            target_vout=_configs['target_vout'])
    else:
        top_1_for_klist = get_topk_rewards_anal(
            collected_surrogate_rewards=np.array(collected_surrogate_rewards),
            cand_states=cand_states, k_list=args.k_list, sim=sim, target_vout=_configs['target_vout'])
    # resync know ground truth
    sim.graph_2_reward = tmp_graph_to_reward
    return top_1_for_klist, time.time() - start_time


def assign_retrain_paramters(debug_AL, factory, query_nums, seed, iter_idx):
    if debug_AL:
        factory.sample_ratio, factory.epoch, factory.patience = 0.0001, 1, 1000
    else:
        factory.sample_ratio = (0.075 + (sum(query_nums) - 40) / (560 / 0.425)) / 5
        factory.epoch = math.floor(10 + ((sum(query_nums) - 40) / (560 / 48)) / 5)
        factory.patience = math.floor(5 + ((sum(query_nums) - 40) / (560 / 10)) / 5)
    print(f"factory: sp, ep, pa: {factory.sample_ratio}, {factory.epoch}, {factory.patience}"
          f"\n seed and iter: {seed}, {iter_idx}")
    return factory


def write_topks_results_json(result_folder, traj_rows_all, json_suffixs):
    for ind, traj_rows in enumerate(traj_rows_all):
        for _k in traj_rows:
            json.dump(traj_rows[_k],
                      open(f"{result_folder}{json_suffixs[ind]}_{args.output}-{str(_k)}.json", 'w'))


def get_surrogate_rewards(sim, cand_states, method):
    ensemble_start_time = time.time()
    if method == 'seq_ensemble':
        ensemble_infos = sim.sequential_generate_ensemble_infos(cand_states)
        collected_surrogate_rewards = ensemble_infos[0]
    elif method == 'batch_ensemble':
        ensemble_infos = sim.batch_generate_ensemble_infos(cand_states)
        collected_surrogate_rewards = ensemble_infos
        print(f"calculate ensemble info time:{time.time() - ensemble_start_time}")
    elif method == 'single':
        collected_surrogate_rewards = []
        for state in cand_states:
            sim.set_state(None, None, state)
            collected_surrogate_rewards.append(sim.get_no_sweep_reward())
            ensemble_infos = None
    ensemble_info_time = time.time() - ensemble_start_time
    return collected_surrogate_rewards, ensemble_infos, ensemble_info_time


def gp_reward_uct_exp(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    tmp_eff_models, tmp_vout_models = None, None
    if args.model == 'simulator' or args.model == 'analytics':
        from topo_envs.softwareSim import SoftwareSimulatorFactory
        factory = SoftwareSimulatorFactory()
        output_name = args.output or 'simulator'
    elif args.model == 'transformer':
        from topo_envs.TransformerRewardSim import TransformerRewardSimFactory
        import transformer_SVGP.transformer_config
        # 5_05_vout_9.pt.chkpt
        print(dir)
        training_data = init_training_data(args)
        factory = TransformerRewardSimFactory(
            # eff_model_files=os.path.join(dir, args.eff_model),
            eff_model_file=os.path.join(dir, args.eff_model),  # query_arguments set eff_model from eff_model_seed
            vout_model_file=os.path.join(dir, args.vout_model),
            eff_model_files=[os.path.join(dir, eff_model) for eff_model in args.eff_models],
            # query_arguments set eff_model from eff_model_seed
            vout_model_files=[os.path.join(dir, vout_model) for vout_model in args.vout_models],
            vocab_file=os.path.join(dir, args.vocab),
            dev_file=os.path.join(dir, args.dev_file),
            test_file=os.path.join(dir, args.test_file),
            device=device,
            training_data=training_data,
            eff_model_seed=args.eff_model_seed,
            vout_model_seed=args.vout_model_seed,
            eff_model_seeds=args.eff_model_seeds,
            vout_model_seeds=args.vout_model_seeds,
            epoch=args.epoch,
            patience=args.patience,
            sample_ratio=args.sample_ratio)

        output_name = args.output or 'transformer'
        tmp_eff_models = copy.deepcopy(factory.eff_models)
        tmp_vout_models = copy.deepcopy(factory.vout_models)
    else:
        raise Exception('unknown model ' + args.model)

    seed_range = range(args.seed_range[0], args.seed_range[1])
    result_folder = 'Merged_Results/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + \
                    '-' + str(os.getpid()) + '-'
    mkdir('Merged_Results/')

    # construct config object for UCT
    from UCT_5_UCB_unblc_restruct_DP_v1.config import uct_configs
    _configs = update_configs_with_args(uct_configs, args)
    seed_idx = 0
    pre_traj_rows = init_output_rows(k_list=args.k_list, traj_list=args.traj_list, seed_range=seed_range)
    post_traj_rows = init_output_rows(k_list=args.k_list, traj_list=args.traj_list, seed_range=seed_range)
    prop_traj_rows = init_output_rows(k_list=args.prop_top_ratios, traj_list=args.traj_list, seed_range=seed_range)
    candidate_lengths, avg_distribution_raw_data, inter_num = [], {}, len(args.traj_list[0])
    random_sampling_csv = [[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]]
    for i in range(inter_num):
        avg_distribution_raw_data[i] = []

    for seed in seed_range:
        for traj in args.traj_list:
            logging.info('random seed ' + str(seed))
            feed_random_seeds(seed)
            global_graph_to_reward = {}
            # model needs to be reset for different random seeds
            if args.model == 'transformer':
                factory.reset_model()
            # initially, set uct trees to be None
            duration, query_num, query_for_update, start_time, cand_states, queried_state_keys, uct_tree_lists = \
                0, 0, 0, time.time(), [], set(), [None] * args.replan_times
            # if args.model == 'transformer':
            #     queried_state_keys = set(init_trained_data_keys(training_data.data))

            for inter_idx, traj_num in enumerate(traj):
                logging.info('traj num ' + str(traj_num))
                # since we have multiple replan times, will use the average query num
                if not args.cumulate_candidates:
                    cand_states = []
                # query_nums records the query of each iteration
                query_nums = []
                # the reward function may have updated after active learning
                if args.model == 'transformer':
                    factory.reset_surrogate_table_with_graph_to_reward(global_graph_to_reward, _configs['target_vout'])
                sim_init, inf_time, inf_count = factory.get_sim_init(), 0, 0

                for replan_time in range(args.replan_times):
                    # TODO just for debug
                    feed_random_seeds(seed)
                    info = run_uct(Sim=sim_init, traj=traj_num, configs=_configs,
                                   # uct_tree_list=uct_tree_lists[replan_time] if args.reuse_tree_after_al else None)
                                   uct_tree_list=uct_tree_lists[replan_time] if args.reuse_tree_after_al else None,
                                   keep_uct_tree=args.reuse_tree_after_al)

                    # info['state_list'] is the previous way to get states visited in UCT, but only the states in the nodes,
                    # not the ones visited in the roll-out. This way of getting states is no longer used.
                    # cand_states += info['state_list']
                    uct_tree_lists[replan_time], sim = info['uct_tree_list'], info['sim']
                    cand_states += [topo[0] for topo in sim.no_isom_seen_state_list]
                    query_nums.append(info['query_num'])

                    inf_time += sim.new_query_time
                    inf_count += sim.new_query_counter

                # TODO external hash table is not implemented?
                # reward_hash.update(sim.all_graph_2_reward)
                # random_sampling_results, random_sampling_results_mean = model_random_sampling_statistic(sim)
                # random_sampling_csv.append(random_sampling_results_mean)

                average_query_num_this_iter = int(np.mean(query_nums))

                if args.cumulate_candidates or args.reuse_tree_after_al:
                    query_num += average_query_num_this_iter
                else:
                    query_num = average_query_num_this_iter

                if args.model == 'transformer' and args.save_realR:  # the graph_to_reward saves the
                    sim.graph_2_reward.update(global_graph_to_reward)
                candidate_lengths.append(len(cand_states))
                retrain_duration, effi_early_stop, vout_early_stop, epoch_i_eff, epoch_i_vout = 0, 0, 0, 0, 0

                logging.info('query number this iteration ' + str(average_query_num_this_iter))
                logging.info('state number ' + str(len(cand_states)))

                # get collected_surrogate_rewards
                ensemble_infos = None
                if args.random_top:
                    collected_surrogate_rewards = [random.random() for _ in cand_states]
                    ensemble_infos, ensemble_info_time = None, 0
                else:
                    collected_surrogate_rewards, ensemble_infos, ensemble_info_time = \
                        get_surrogate_rewards(sim=sim, cand_states=cand_states, method='seq_ensemble')

                before_query = len(sim.graph_2_reward)
                tmp_graph_to_reward = copy.deepcopy(sim.graph_2_reward)  # checkpoint of the know ground truth
                # topks for pre
                if args.model == 'transformer' or args.model == 'gnn':
                    print(f"before top retraing query times: {args.query_times}, "
                          f"simulation reward length: {len(sim.graph_2_reward)}")
                    top_1_for_klist = get_topk_rewards_surrogate_model(
                        collected_surrogate_rewards=np.array(collected_surrogate_rewards),
                        cand_states=cand_states, k_list=args.k_list, sim=sim,
                        count_pre_simulated=True, target_vout=_configs['target_vout'])
                else:
                    top_1_for_klist = get_topk_rewards_anal(
                        collected_surrogate_rewards=np.array(collected_surrogate_rewards),
                        cand_states=cand_states, k_list=args.k_list, sim=sim, target_vout=_configs['target_vout'])

                top_query = len(sim.graph_2_reward) - before_query
                duration = time.time() - start_time
                for k, top_1 in top_1_for_klist.items():
                    pre_traj_rows[k][inter_idx][seed_idx].extend(
                        [seed, query_num, top_1['reward'], duration, traj_num, retrain_duration, top_1['query'],
                         len(sim.graph_2_reward), get_observed_topology_count(sim.graph_2_reward), query_for_update,
                         sim.number_of_calling_simulator if args.model == 'transformer' else 0, ensemble_info_time])

                # # topks for prop, if sweep, use top circuit topology,
                prop_top_k_list = [math.ceil(top_ratio * query_num) if not sim.configs_['sweep'] else
                                   math.ceil(top_ratio * query_num / len(sim.candidate_params))
                                   for top_ratio in args.prop_top_ratios]

                red_prop_top_k_list = [j for m, j in enumerate(prop_top_k_list) if j not in prop_top_k_list[:m]]
                sim.graph_2_reward = tmp_graph_to_reward  # resync know ground truth

                if args.model == 'transformer' or args.model == 'gnn':
                    tmp_after_pre_graph_to_reward = copy.deepcopy(sim.graph_2_reward)

                    prop_top_1_for_top_ratios = get_topk_rewards_surrogate_model(  # prop top ratios results
                        collected_surrogate_rewards=np.array(collected_surrogate_rewards),
                        cand_states=cand_states, k_list=red_prop_top_k_list, sim=sim,
                        count_pre_simulated=True, target_vout=_configs['target_vout'])
                    sim.graph_2_reward.update(tmp_after_pre_graph_to_reward)  # resync know ground truth
                else:  # prop topks for anal
                    prop_top_1_for_top_ratios = get_topk_rewards_anal(
                        collected_surrogate_rewards=np.array(collected_surrogate_rewards),
                        cand_states=cand_states, k_list=red_prop_top_k_list, sim=sim,
                        target_vout=_configs['target_vout'])
                for ratio_idx, _top_ratio in enumerate(args.prop_top_ratios):
                    prop_traj_rows[_top_ratio][inter_idx][seed_idx].extend(
                        [seed, query_num, prop_top_1_for_top_ratios[prop_top_k_list[ratio_idx]]['reward'],
                         duration, traj_num, retrain_duration,
                         prop_top_1_for_top_ratios[prop_top_k_list[ratio_idx]]['query'],
                         len(sim.graph_2_reward), get_observed_topology_count(sim.graph_2_reward),
                         query_for_update, sim.number_of_calling_simulator if args.model == 'transformer' else 0,
                         ensemble_info_time])

                args.query_times = math.ceil(sum(query_nums) * args.retrain_query_ratio)
                query_for_update += args.query_times
                random_query_times = 0

                retrain_start_time = time.time()
                before_query = len(sim.graph_2_reward)

                if (args.model == 'transformer' or args.model == 'gnn') and args.update_rewards:
                    if not args.cumulate_candidates: queried_state_keys = []
                    factory = assign_retrain_paramters(args.debug_AL, factory, query_nums, seed, iter_idx=inter_idx)
                    # if args.model == 'transformer' or args.model == 'gnn':
                    sim.graph_2_reward = tmp_graph_to_reward  # resync know ground truth

                    surrogate_rewards, effi_early_stop, vout_early_stop, queried_state_keys, epoch_i_eff, epoch_i_vout \
                        = query_update(state_list=cand_states, ensemble_infos=ensemble_infos,
                                       queried_state_keys=queried_state_keys, sim=sim, factory=factory,
                                       update_gp=args.update_gp if inter_idx < len(traj) - 1 else False,
                                       # don't need to update in the final step
                                       query_times=args.query_times, strategy=args.AL_strategy)
                retrain_duration = time.time() - retrain_start_time
                update_query = len(sim.graph_2_reward) - before_query  # calculate how many queries in current AL
                print(
                    f"after retraing query times: {args.query_times}, simulation reward length: {len(sim.graph_2_reward)}")
                # cumulative_queried_states.extend(queried_states)
                if (args.model == 'transformer' or args.model == 'gnn') and args.save_realR:
                    global_graph_to_reward.update(sim.graph_2_reward)

                if (args.model in ['transformer', 'gnn']) and args.reuse_tree_after_al and args.recompute_tree_rewards:
                    # recompute rewards in the tree
                    for uct_tree_list in uct_tree_lists:
                        for uct_tree in uct_tree_list:
                            recompute_tree_rewards(uct_tree, sim)

                # update post topks rewards
                top_1_for_klist, duration = \
                    get_top_for_klist(_args=args, collected_surrogate_rewards=collected_surrogate_rewards,
                                      cand_states=cand_states, sim=sim, _configs=_configs, start_time=start_time)
                for k, top_1 in top_1_for_klist.items():
                    post_traj_rows[k][inter_idx][seed_idx].extend(
                        [seed, query_num, top_1['reward'], duration, traj_num, retrain_duration,
                         top_1['query'], len(sim.graph_2_reward), get_observed_topology_count(sim.graph_2_reward),
                         update_query, float(epoch_i_eff), float(epoch_i_vout)])

                print(f"post query times: {args.query_times}, simulation reward length: {len(sim.graph_2_reward)}")

            # write the top k's results to the json files
            write_topks_results_json(result_folder, [pre_traj_rows, post_traj_rows, prop_traj_rows],
                                     ['pre', 'post', 'prop'])

            # for k in post_traj_rows:
            #     json.dump(post_traj_rows[k], open(result_folder + 'post_' + args.output + '-' + str(k) + '.json', 'w'))
            # for prop_ratio in prop_traj_rows:
            #     json.dump(prop_traj_rows[prop_ratio],
            #               open(result_folder + 'prop_' + args.output + '-' + str(prop_ratio) + '.json', 'w'))
            gc.collect()
        seed_idx += 1

    # write the results to the csv files
    save_results_to_csv(pre_traj_rows, result_folder, 'pre_' + output_name)
    save_results_to_csv(post_traj_rows, result_folder, 'post_' + output_name)
    save_results_to_csv(prop_traj_rows, result_folder, 'prop_' + output_name)

    with open(result_folder + output_name + '-random_sampling' + '.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(random_sampling_csv)

    print(candidate_lengths)

    os.system('mkdir ./Merged_Results/' + str(os.getpid()))
    os.system('cp -r ./Merged_Results/*-' + str(os.getpid()) + '-* ./Merged_Results/' + str(os.getpid()))
    os.system('cp -r ./Merged_Results/*' + str(os.getpid()) + '*.png ./Merged_Results/' + str(os.getpid()))


if __name__ == '__main__':
    args = get_args()
    import torch

    logging.basicConfig(filename=args.output + '.log',
                        filemode='w',
                        level=logging.INFO)
    # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #     for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #         args.eff_model_seed, args.vout_model_seed = i, j
    #         #
    #         #         # args.eff_model = 'transformer_SVGP/save_model/5_05/5_05_eff_' + str(
    #         #         #     args.eff_model_seed) + '.pt'
    #         #         # args.vout_model = 'transformer_SVGP/save_model/5_05/5_05_vout_' + str(
    #         #         #     args.vout_model_seed) + '.pt'
    #         #         # args.eff_model = 'transformer_SVGP/lstm_model/5_05_fix_eff_' + str(args.eff_model_seed) + '.pt'
    #         #         # args.vout_model = 'transformer_SVGP/lstm_model/5_05_fix_vout_' + str(args.vout_model_seed) + '.pt'
    #         args.eff_model = args.model_prefix + '_eff_fix_comp_06_' + str(args.eff_model_seed) + '.pt'
    #         args.vout_model = args.model_prefix + '_vout_fix_comp_06_' + str(args.vout_model_seed) + '.pt'
    #         # args.eff_model = 'transformer_SVGP/lstm_model_01/5_05_fix_eff_01_' + str(args.eff_model_seed) + '.pt'
    #         # args.vout_model = 'transformer_SVGP/lstm_model_01/5_05_fix_vout_01_' + str(args.vout_model_seed) + '.pt'
    #
    #         gp_reward_uct_exp(args)
    gp_reward_uct_exp(args)

    # reward calculate---------------{'efficiency': 0.870614956398022, 'output_voltage': 183.3787097490262}, 0.7507610722438551
    # reward calculate---------------{'efficiency': 0.7814244536468952, 'output_voltage': -204.66270751924748}, 0.7723692834422491
