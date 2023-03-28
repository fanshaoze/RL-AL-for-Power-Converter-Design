import argparse
import csv
import json
import math
import random

import torch

from transformer_utils import evaluate_model, transpose, evaluate_eff_vout_model, compute_batch_reward
from matplotlib import pyplot as plt
import statistics
from util import compute_rse
import numpy as np




# vout_range_bounds = [-300, -200, -100, -50, 0, 25, 45, 55, 75, 90, 110, 200, 300 + math.pow(0.1, 10)]
# fix_data_0.075_extra__extra_reward_plot-0
# [0.07244758717715741, 0.05257325079292059, 0.06335364439221368, 0.8854911154610704, 0.04606381203601536, 0.7835108723691598, 1.6965853481104376]

def vote_and_count(values, range_list):
    vote_hash = {}
    for _range in range_list: vote_hash[_range] = []
    for idx, v in enumerate(values):
        check_in_range_flag = 0
        for _range, votes in vote_hash.items():
            if _range[0] <= v < _range[1]:
                votes.append(idx)
                check_in_range_flag = 1
                break
        if not check_in_range_flag:
            print(f"prediction not in any range: {v}")
            exit(0)
    max_vote_count = -1
    for _range, votes in vote_hash.items():
        if len(votes) > max_vote_count:
            max_vote_count = len(votes)
            voted_range = _range
    return vote_hash, voted_range


def ensemble_information_generation(data_preds, range_bound_list):
    """
    avg of the max vote range
    :param data_preds:
    :param range_bound_list:
    :return:
    """
    ensemble_predictions, vote_count = [], []
    uncertainty_counts, uncertainty_stds = [], []
    range_list = []  # range list: [(low_edge<=, <high_edge)], <= low as eff have 0 as prediction
    for i in range(len(range_bound_list) - 1):
        range_list.append((range_bound_list[i], range_bound_list[i + 1]))
    for predictions in data_preds:
        vote_hash, voted_range = vote_and_count(predictions, range_list)
        ensemble_predictions.append(statistics.mean([predictions[idx] for idx in vote_hash[voted_range]]))
        # ensemble_prediction.append(statistics.median([predictions[idx] for idx in vote_hash[voted_range]]))
        vote_count.append(len(vote_hash[voted_range]))
        uncertainty_counts.append(sum([len(vote_hash[_range]) for _range in vote_hash if _range != voted_range]))
        # if predictions
        uncertainty_stds.append(statistics.stdev([float(_pred) for _pred in predictions]))
    return ensemble_predictions, vote_count, uncertainty_counts, uncertainty_stds


def get_tops(gt, info, tops):
    top_results = []
    for top in tops:
        top_results.append(max([gt[query_ind] for query_ind in
                                sorted([ind for ind in range(len(info))],
                                       key=lambda i: info[i], reverse=True)[:top]]))
    return top_results


def avg_prediction_generation(data_preds):
    """
    avg of the max vote range
    :param data_preds:
    :return:
    """
    ensemble_prediction = []
    for predictions in data_preds:
        ensemble_prediction.append(statistics.mean(predictions))
    return ensemble_prediction


def evaluate_trained_transformer_reward(data_file, use_gp, eff_model=None, vout_model=None, eff_seeds=None,
                                        vout_seeds=None,
                                        output_file=None, training_ratio=0.6, print_distribution=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    vocab_file = 'compwise_vocab.json'
    # eff_model_prefix = f'lstm_model/5_all_lstm_gp_eff_16_{str(training_ratio)}_'
    # vout_model_prefix = f'lstm_model/5_all_lstm_gp_vout_16_{str(training_ratio)}_'
    eff_model_prefix = f'lstm_model/5_all_lstm_eff_16_{str(training_ratio)}_'
    vout_model_prefix = f'lstm_model/5_all_lstm_vout_16_{str(training_ratio)}_'

    dataset = json.load(open(data_file, 'r'))
    # add ensemble_head, ensemble_index, ensemble_gt_rewards to ensemble_predictions
    ensemble_predictions = [[i for i in range(len(dataset))],
                            [datum['reward'] for datum in dataset]]
    output_file = output_file or data_file

    with open(f"{output_file}_tops_extra.csv", 'w') as f:
        # with open(f"{output_file}_tops.csv", 'w') as f:
        csv_writer = csv.writer(f)
        # header = ['eff model', 'vout model', '1', '10', '50', '100']
        header = ['eff model', 'vout model', 'eff rse', 'vout rse', '1', '10', '50', '100']
        csv_writer.writerow(header)

        eff_seeds = 1 if eff_model else eff_seeds
        vout_seeds = 1 if vout_model else vout_seeds

        for eff_seed in range(eff_seeds):
            for vout_seed in range(vout_seeds):
                print(eff_seed, vout_seed)
                predictions, ground_truths, model_perform, pred_effs, pred_vouts, eff_rse, vout_rse = \
                    evaluate_model(
                        eff_model=eff_model or eff_model_prefix + str(eff_seed) + '.pt',
                        vout_model=vout_model or vout_model_prefix + str(vout_seed) + '.pt',
                        vocab=vocab_file,
                        data=data_file,
                        device=device,
                        use_gp=use_gp,
                    )

                # plot surrogate vs gt
                if print_distribution:
                    plt.scatter(ground_truths, predictions, s=5)
                    plt.xlabel('Ground Truth Reward')
                    plt.ylabel('Surrogate Reward')
                    plt.ylim(0, 1)
                    plt.savefig(f"{output_file}.{eff_seed}-{vout_seed}_extra.png", dpi=1200, format="png")
                    # plt.savefig(f"{output_file}.{eff_seed}-{vout_seed}.png", dpi=1200, format="png")
                    plt.close()

                ensemble_predictions.append(predictions)

                # write top-k results
                csv_writer.writerow([eff_seed, vout_seed, eff_rse, vout_rse] + list(model_perform.values()))
    f.close()
    with open(f"{output_file}_ensemble_extra.csv", 'w') as f:
        # with open(f"{output_file}_ensemble.csv", 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['index', 'eff G', 'vout G', 'reward G'])
        csv_writer.writerows(transpose(ensemble_predictions))
    f.close()


def get_gts(dataset):
    gt_effs, gt_vouts, gt_rewards = [], [], []
    for datum in dataset:
        if datum['valid']:
            gt_effs.append(datum['eff'])
            gt_vouts.append(datum['vout'])
            gt_rewards.append(datum['reward'])
        else:
            gt_effs.append(0)
            gt_vouts.append(-300)
            gt_rewards.append(0)
    return gt_effs, gt_vouts, gt_rewards


def select_inds(sort_inds, select_count, selected_inds):
    """
    select select_count inds from the sort_inds without overlap with selected_ind
    :return:
    """
    query_indices = []
    for ind in sort_inds:
        if select_count > 0:
            if ind not in selected_inds:
                query_indices.append(ind)
                select_count -= 1
        else:
            break
    return query_indices


def select_using_uncertainty(query_cand_indices, eff_uncertainty_var, vout_uncertainty_var, un_count, un_eff_ratio):
    eff_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: eff_uncertainty_var[i], reverse=True),
        select_count=int(un_count * un_eff_ratio),
        selected_inds=[])
    vout_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: vout_uncertainty_var[i], reverse=True),
        select_count=un_count - int(un_count * un_eff_ratio),
        selected_inds=eff_query_indices)
    return eff_query_indices + vout_query_indices


def select_using_prediction(query_cand_indices, reward_ensemble_predictions, pred_count, low_pred_ratio, selected_inds):
    low_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: reward_ensemble_predictions[i], reverse=False),
        select_count=int(pred_count * low_pred_ratio),
        selected_inds=selected_inds)
    high_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: reward_ensemble_predictions[i], reverse=True),
        select_count=pred_count - int(pred_count * low_pred_ratio),
        selected_inds=selected_inds + low_query_indices)
    return high_query_indices + low_query_indices


def hybrid_query_strategy(query_cand_indices, reward_ensemble_predictions,
                          eff_uncertainty_var, vout_uncertainty_var, retrain_query_count, un_ratio, un_eff_ratio,
                          low_pred_ratio):
    query_indices = []
    un_count, pred_count = int(retrain_query_count * un_ratio), retrain_query_count - int(
        retrain_query_count * un_ratio)
    query_indices += select_using_uncertainty(query_cand_indices, eff_uncertainty_var, vout_uncertainty_var,
                                              un_count, un_eff_ratio)
    query_indices += select_using_prediction(query_cand_indices, reward_ensemble_predictions,
                                             pred_count, low_pred_ratio, selected_inds=query_indices)
    return query_indices


def sample_based_tops(gt_rewards, reward_ensemble_predictions, eff_uncertainty_var, vout_uncertainty_var, _top_ratios,
                      top_queries, un_ratio, un_eff_ratio, low_pred_ratio):
    run_times, results = 1000, {}
    for query_times in top_queries:
        results[query_times] = {}
        for top in _top_ratios:
            results[query_times][top] = 0
        for run_time in range(run_times):
            print(query_times, ':', run_time)
            query_cand_indices = random.sample([i for i in range(len(gt_rewards))], query_times)
            for top_ratio in _top_ratios:
                query_indices = \
                    hybrid_query_strategy(query_cand_indices, reward_ensemble_predictions,
                                          eff_uncertainty_var, vout_uncertainty_var,
                                          retrain_query_count=int(top_ratio * query_times),
                                          un_ratio=un_ratio, un_eff_ratio=un_eff_ratio, low_pred_ratio=low_pred_ratio)
                results[query_times][top_ratio] += (max([gt_rewards[i] for i in query_indices]) -
                                                    results[query_times][top_ratio]) / (
                                                           run_time + 1)  # incremental average
    return results


def plot_results(results, tops, file_name):
    """
    :param results: {query:{top:avg_reward}}
    :return:
    """
    results_queries = [q for q in results]
    for top_ratio in tops:
        rewards = [results[q][top_ratio] for q in results_queries]
        plt.plot(results_queries, rewards, alpha=10 * top_ratio, linewidth=1, color='r', label=str(top_ratio))
    plt.legend(loc='best')
    plt.savefig(file_name + '.png', dpi=1200, format="png")


queries = [300, 500, 1000, 2000, 4000]
max_top_ratio = 0.1

top_ratios = [i * 0.01 for i in range(1, int(max_top_ratio / 0.01) + 1)]

eff_range_bounds = [0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + math.pow(0.1, 10)]
# + [0.8 + 0.2 * x for x in range(10)] + [1.0 + math.pow(0.1, 10)]
vout_range_bounds = [-300, -200, -100, -50, 0, 25, 45, 55, 75, 90, 110, 200, 300 + math.pow(0.1, 10)]


def evaluate_ensemble_trained_transformer(args, data_file, use_gp, eff_model=None, vout_model=None,
                                          eff_seeds=None, vout_seeds=None, eff_pred_file='pred_eff',
                                          vout_pred_file='pred_vout', output_file=None,
                                          training_ratio=0.6, print_distribution=False,
                                          target_vout=50):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    vocab_file = 'compwise_vocab.json'
    # eff_model_prefix = f'lstm_model/5_all_lstm_gp_eff_16_{str(training_ratio)}_'
    # vout_model_prefix = f'lstm_model/5_all_lstm_gp_vout_16_{str(training_ratio)}_'
    # eff_model_prefix = f'lstm_model/5_all_lstm_gp_eff_16_{str(training_ratio)}_extra_'
    # vout_model_prefix = f'lstm_model/5_all_lstm_gp_vout_16_{str(training_ratio)}_extra_'
    eff_model_prefix = f'lstm_model/5_all_lstm_eff_16_{str(training_ratio)}_'
    vout_model_prefix = f'lstm_model/5_all_lstm_vout_16_{str(training_ratio)}_'

    dataset = json.load(open(data_file, 'r'))
    # add ensemble_head, ensemble_index, ensemble_gt_rewards to ensemble_predictions
    gt_effs, gt_vouts, gt_rewards = get_gts(dataset)
    ensemble_predictions = [[i for i in range(len(dataset))], gt_effs, gt_vouts, gt_rewards]

    output_file = output_file or data_file

    eff_seeds = 1 if eff_model else eff_seeds
    vout_seeds = 1 if vout_model else vout_seeds

    data_pred_effs = []
    data_pred_vouts = []
    eff_avg_RSE, vout_avg_RSE = 0, 0

    eff_preds_from_file = json.load(open(eff_pred_file, 'r'))
    for eff_seed in range(eff_seeds):
        # eff_seed=17
        if str(eff_seed) in eff_preds_from_file:
            eff_preds, eff_rse = np.array(eff_preds_from_file[str(eff_seed)]['preds']), \
                                 eff_preds_from_file[str(eff_seed)]['rse']
        else:
            eff_preds, eff_rse = evaluate_eff_vout_model(
                _model=eff_model or eff_model_prefix + str(eff_seed) + '.pt', vocab=vocab_file,
                data=data_file, device=device, use_gp=use_gp, target='eff', batch_size=args.batch_size)
            eff_preds_from_file[str(eff_seed)] = {'preds': [float(_pred) for _pred in eff_preds], 'rse': float(eff_rse)}
        eff_avg_RSE += eff_rse
        ensemble_predictions.append(eff_preds)
        data_pred_effs.append(eff_preds)
    # json.dump(eff_preds_from_file, open(eff_pred_file, 'w'))
    data_pred_effs = transpose(data_pred_effs)
    eff_ensemble_predictions, eff_vote_counts, eff_uncertainty_counts, eff_uncertainty_vars = \
        ensemble_information_generation(data_preds=data_pred_effs, range_bound_list=eff_range_bounds)
    eff_avg_vote_count = statistics.mean(eff_vote_counts)

    vout_preds_from_file = json.load(open(vout_pred_file, 'r'))
    for vout_seed in range(vout_seeds):
        # vout_seed = 7
        if str(vout_seed) in vout_preds_from_file:
            vout_preds, vout_rse = np.array(vout_preds_from_file[str(vout_seed)]['preds']), \
                                   vout_preds_from_file[str(vout_seed)]['rse']
        else:
            vout_preds, vout_rse = evaluate_eff_vout_model(
                _model=vout_model or vout_model_prefix + str(vout_seed) + '.pt', vocab=vocab_file,
                data=data_file, device=device, use_gp=use_gp, target='vout', batch_size=args.batch_size)
            vout_preds_from_file[str(vout_seed)] = {'preds': [float(_pred) for _pred in vout_preds],
                                                    'rse': float(vout_rse)}
        vout_avg_RSE += vout_rse
        ensemble_predictions.append(vout_preds)
        data_pred_vouts.append(vout_preds)
    # json.dump(vout_preds_from_file, open(vout_pred_file, 'w'))
    # exit(0)
    data_pred_vouts = transpose(data_pred_vouts)
    print(data_pred_vouts)
    vout_ensemble_predictions, vout_vote_counts, vout_uncertainty_counts, vout_uncertainty_vars = \
        ensemble_information_generation(data_preds=data_pred_vouts, range_bound_list=vout_range_bounds)
    vout_avg_vote_count = statistics.mean(vout_vote_counts)

    reward_ensemble_predictions = compute_batch_reward(eff_ensemble_predictions, vout_ensemble_predictions,
                                                       target_vout=target_vout)

    if args.sample_result:
        results = sample_based_tops(gt_rewards, reward_ensemble_predictions, eff_uncertainty_vars,
                                    vout_uncertainty_vars,
                                    _top_ratios=top_ratios, top_queries=queries,
                                    un_ratio=0.2, un_eff_ratio=0.1, low_pred_ratio=0.1)
        plot_results(results, top_ratios, file_name='un-02-eff-01-low-01')
        with open(f"un-02-eff-01-low-01.csv", 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(queries)
            csv_writer.writerows(transpose([list(results[q].values()) for q in queries]))
        exit(0)

    eff_avg_RSE, vout_avg_RSE = eff_avg_RSE / eff_seeds, vout_avg_RSE / vout_seeds,

    eff_ensemble_rse = compute_rse(pred=np.clip(np.array(eff_ensemble_predictions), 0., 1.),
                                   y=np.clip(np.array(gt_effs), 0., 1.))
    vout_ensemble_rse = compute_rse(pred=np.clip((np.array(vout_ensemble_predictions) + 300.) / 600., 0., 1.),
                                    y=np.clip((np.array(gt_vouts) + 300.) / 600., 0., 1.))
    reward_ensemble_rse = compute_rse(pred=np.clip(np.array(reward_ensemble_predictions), 0., 1.),
                                      y=np.clip(np.array(gt_rewards), 0., 1.))

    plt.scatter(gt_rewards, reward_ensemble_predictions, s=1)
    plt.xlabel('Ground Truth Reward')
    plt.ylabel('Surrogate Reward')
    plt.ylim(0, 1)
    plt.savefig(f"{output_file}_extra_reward_plot.png", dpi=1200, format="png")
    # plt.savefig(f"{output_file}.{eff_seed}-{vout_seed}.png", dpi=1200, format="png")
    plt.close()

    ensemble_predictions += [eff_ensemble_predictions, vout_ensemble_predictions, reward_ensemble_predictions]
    ensemble_result_heads = ['eff avg RSE', 'vout avg RSE', 'ensemble eff RSE', 'eff clustering rate',
                             'ensemble vout RSE', 'vout clustering rate', 'reward RSE']
    ensemble_results = [eff_avg_RSE, vout_avg_RSE,
                        eff_ensemble_rse, eff_avg_vote_count / eff_seeds,
                        vout_ensemble_rse, vout_avg_vote_count / vout_seeds, reward_ensemble_rse]
    print(f"{output_file}_ensemble_extra.csv", "\n", ensemble_result_heads, "\n", ensemble_results)

    with open(f"{output_file}_ensemble_extra.csv", 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['index', 'eff G', 'vout G', 'reward G'] +
                            [f'eff {eff_seed}' for eff_seed in range(eff_seeds)] +
                            [f'vout {vout_seed}' for vout_seed in range(vout_seeds)] +
                            ['eff P', 'vout P', 'reward P', 'RSEs'])
        csv_writer.writerows(transpose(ensemble_predictions))
        csv_writer.writerows([ensemble_result_heads, ensemble_results])
    f.close()


def get_transformer_args(arg_list=None):
    parser = argparse.ArgumentParser()

    # used for auto_train.py to split training data
    parser.add_argument('-sample_result', action='store_true', default=False)

    parser.add_argument('-eff_model', type=str, default=None, help='eff model')
    parser.add_argument('-vout_model', type=str, default=None, help='vout model')
    parser.add_argument('-eff_seeds', type=int, default=7, help='eff seeds from 0 to eff_seeds')
    parser.add_argument('-vout_seeds', type=int, default=13, help='vout seeds from 0 to vout_seeds')
    parser.add_argument('-output_file', type=str, default='fix_data_0.075_extra_7_13_', help='output file')
    parser.add_argument('-data_file', type=str, default='dataset_5_cleaned_label_train_0.075_sample.json',
                        help='data_file')
    # parser.add_argument('-data_file', type=str, default='debug_test.json', help='data_file')
    parser.add_argument('-eff_pred_file', type=str, default='fix_data_0.075_pred_eff.json', help='saved eff pred file')
    parser.add_argument('-vout_pred_file', type=str, default='fix_data_0.075_pred_vout.json', help='saved vout pred file')

    parser.add_argument('-print_distribution', action='store_true', default=False)
    parser.add_argument('-target_vout', type=float, default=50, help='target vout')

    parser.add_argument('-batch_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # TODO: use top reward to firstly filter some unreasonable models, then ensemble

    args = get_transformer_args()

    evaluate_ensemble_trained_transformer(
        args=args, eff_model=args.eff_model, vout_model=args.vout_model,
        eff_seeds=args.eff_seeds, vout_seeds=args.vout_seeds,
        output_file=args.output_file, data_file=args.data_file, use_gp=True,
        eff_pred_file=args.eff_pred_file, vout_pred_file=args.vout_pred_file,
        training_ratio=0.075, print_distribution=False, target_vout=50)
