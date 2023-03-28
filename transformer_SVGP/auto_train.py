import itertools
import json
import logging
import os
import random
import sys
import numpy as np
import wandb
from tqdm import tqdm

from transformer_args import get_transformer_args
from util import feed_random_seeds, distribution_plot

sys.path.append(os.path.join(sys.path[0], '../topo_data_util/'))
from train import main as train_fn
from topo_analysis.topoGraph import TopoGraph
from GetReward import calculate_reward
from topo_utils.plot import plot_hist


def parse_json_data(data_path, target_vout=50, select_cond='none', use_log=False):
    """
    Convert the dataset json to formats that can be loaded by transformer

    :param data_path: path of the raw data
    :param target_vout: target output voltage for reward
    :param select_cond: 'max_reward' finds the one with the highest reward;
       'fix_cycle' finds the one with a fixed duty cycle (0.5).
    """
    print('loading from json file')
    raw_data = json.load(open(data_path, 'r'))
    data = []

    print('processing data')
    for name, datum in tqdm(raw_data.items()):
        # get the name of the topology, get rid of the parameter postfix
        # topo_name = '-'.join(name.split('-')[:-1])
        effi = {'efficiency': datum["eff"], 'output_voltage': datum["vout"]}
        reward = calculate_reward(effi, target_vout)

        paths = TopoGraph(node_list=datum['list_of_node'],
                          edge_list=datum['list_of_edge']).find_end_points_paths_as_str()

        # filter
        if select_cond == 'fix_duty':
            if datum['duty_cycle'] != 0.5:
                continue

        if "valid" in datum.keys():
            is_valid = datum["valid"]
        elif "label" in datum.keys():
            is_valid = datum["label"]
        else:
            is_valid = 1  # if no valid label, assume it's valid

        datum = {
            "name": name,
            "list_of_edges": datum['list_of_edge'],
            "list_of_nodes": datum['list_of_node'],
            "paths": paths,
            "eff": datum["eff"] if is_valid else 0.,
            "eff_analytic": datum["eff_analytic"] if is_valid else 0.,
            "vout": datum["vout"] if is_valid else 0.,
            "vout_analytic": datum["vout_analytic"] if is_valid else 0.,
            "duty": datum["duty_cycle"],
            "valid": is_valid,
            "reward": reward
        }
        data.append(datum)

    eff_data = np.clip([datum['eff'] for datum in data], -1, 2)
    plot_hist(eff_data, 'efficiency', f"{data_path}.eff", bins=50, use_log=use_log)

    eff_data = np.clip([datum['eff_analytic'] for datum in data], -1, 2)
    plot_hist(eff_data, 'analytic efficiency', f"{data_path}.eff_analytic", bins=50, use_log=use_log)

    vout_data = np.clip([datum['vout'] for datum in data], -500, 500)
    plot_hist(vout_data, 'vout', f"{data_path}.vout", bins=50, use_log=use_log)

    vout_data = np.clip([datum['vout_analytic'] for datum in data], -500, 500)
    plot_hist(vout_data, 'analytic vout', f"{data_path}.vout_analytic", bins=50, use_log=use_log)

    valid_data = [datum['valid'] for datum in data]
    plot_hist(valid_data, 'is valid', f"{data_path}.valid", bins=10, use_log=use_log)

    return data


def get_topo_name(name):
    return name.split('$')[0]


def merge_extra_good_circuits(smaller_circuit_datasets, _args, good_circuit_threshold):
    """

    :param good_circuit_threshold:
    :param smaller_circuit_datasets:
    :param _args:
    :return:
    """
    extra_data = []
    for data_file in smaller_circuit_datasets:
        print(data_file)
        _data = parse_json_data(data_path=data_file, select_cond=_args.select_cond, use_log=_args.use_log,
                                target_vout=_args.target_vout)
        for dataum in _data:
            if dataum['reward'] > good_circuit_threshold:
                print(str(dataum['reward']) + ' ', end='')
        print(' ')
        extra_data += [dataum for dataum in _data if dataum['reward'] > good_circuit_threshold]
    return extra_data


def split_data(data_path, data, training_ratio, dev_ratio, test_ratio, split_by='data', debug=False, seed=0):
    """
    Split data into training, dev and test sets.
    """
    if split_by == 'data':
        data_size = len(data)
        train_data_size = int(data_size * training_ratio)
        dev_data_size = int(data_size * dev_ratio)
        test_data_size = data_size - train_data_size - dev_data_size if test_ratio == 0 else int(data_size * test_ratio)

        print('total data', data_size)
        print('training size', train_data_size)
        print('dev size', dev_data_size)
        print('test size', test_data_size)

        # randomly permute the data
        random.shuffle(data)
        data_train = data[:train_data_size]
        data_dev = data[train_data_size:train_data_size + dev_data_size]
        data_test = data[train_data_size + dev_data_size:train_data_size + dev_data_size + test_data_size]
    elif split_by == 'topo':
        # dict indexed by topologies
        topo_data_dict = {}
        print('re-indexing by topologies')
        for datum in tqdm(data):
            topo_name = get_topo_name(datum['name'])
            if topo_name not in topo_data_dict.keys():
                topo_data_dict[topo_name] = []
            topo_data_dict[topo_name].append(datum)

        topo_names = list(topo_data_dict.keys())

        data_size = len(topo_names)
        train_data_size = int(data_size * training_ratio)
        dev_data_size = int(data_size * dev_ratio)
        test_data_size = data_size - train_data_size - dev_data_size if test_ratio == 0 else int(data_size * test_ratio)

        if wandb.run is not None:
            wandb.run.summary["total_data_size"] = data_size
            wandb.run.summary["training_data_size"] = train_data_size
            wandb.run.summary["dev_data_size"] = dev_data_size
            wandb.run.summary["test_data_size"] = test_data_size

        print('total number of topologies', data_size)
        print('training size', train_data_size)
        print('dev size', dev_data_size)
        print('test size', test_data_size)

        # randomly permute the data
        random.shuffle(topo_names)
        data_train_names = topo_names[:train_data_size]
        data_dev_names = topo_names[train_data_size:train_data_size + dev_data_size]
        data_test_names = topo_names[train_data_size + dev_data_size:train_data_size + dev_data_size + test_data_size]

        data_train = [topo_data_dict[topo_name] for topo_name in data_train_names]
        # concatenate the lists
        data_train = list(itertools.chain(*data_train))

        data_dev = [topo_data_dict[topo_name] for topo_name in data_dev_names]
        data_dev = list(itertools.chain(*data_dev))

        data_test = [topo_data_dict[topo_name] for topo_name in data_test_names]
        data_test = list(itertools.chain(*data_test))
    else:
        raise Exception(f"unknown split_by {split_by}")

    if debug:
        data_test = data_train

    # get rid of the last element
    file_name = data_path.split('.')[0]

    json.dump(data, open(file_name + '_all.json', 'w'))
    # files are indexed by training ratio
    print(f"saving to {file_name}_*_{training_ratio}_{seed}.json")
    json.dump(data_train, open(f"{file_name}_train_{training_ratio}_{seed}.json", 'w'))
    json.dump(data_dev, open(f"{file_name}_dev_{training_ratio}_{seed}.json", 'w'))
    json.dump(data_test, open(f"{file_name}_test_{training_ratio}_{seed}.json", 'w'))
    return data_train, data_dev, data_test


def splitted_data_exist(data_path, training_ratio, seed):
    file_name = data_path.split('.')[0]

    return os.path.exists(f"{file_name}_train_{training_ratio}_{seed}.json") \
           and os.path.exists(f"{file_name}_dev_{training_ratio}_{seed}.json") \
           and os.path.exists(f"{file_name}_test_{training_ratio}_{seed}.json")


if __name__ == '__main__':
    args = get_transformer_args()
    file_name = args.data.split('.')[0]

    feed_random_seeds(args.seed)

    if args.data is None:
        raise Exception('args.data not provided.')

    if args.use_log:
        wandb.init(project="surrogate_model",
                   name=file_name,
                   config=vars(args))
    if not args.data_seed:
        print('not using fixed dataset')
        args.data_seed = args.seed
    if not splitted_data_exist(data_path=args.data, training_ratio=args.train_ratio, seed=args.data_seed):
        # add 3,4 good circuits
        data = merge_extra_good_circuits(smaller_circuit_datasets=args.extra_datasets, _args=args,
                                         good_circuit_threshold=args.circuit_threshold)
        data = data + parse_json_data(data_path=args.data, select_cond=args.select_cond, use_log=args.use_log,
                                      target_vout=args.target_vout)

        data_train, data_dev, data_test = split_data(data=data,
                                                     data_path=args.data,
                                                     training_ratio=args.train_ratio,
                                                     dev_ratio=args.dev_ratio,
                                                     test_ratio=args.test_ratio,
                                                     split_by=args.split_by,
                                                     debug=args.debug,
                                                     seed=args.seed)
        data_train_file = f"{file_name}_train_{args.train_ratio}_{args.seed}.json"
        data_dev_file = f"{file_name}_dev_{args.train_ratio}_{args.seed}.json"
        data_test_file = f"{file_name}_test_{args.train_ratio}_{args.seed}.json"
        exit(0)

    else:
        print('keep using existing splitted data')
        data_train_file = f"{file_name}_train_{args.train_ratio}_{args.data_seed}.json"
        data_dev_file = f"{file_name}_dev_{args.train_ratio}_{args.data_seed}.json"
        data_test_file = f"{file_name}_test_{args.train_ratio}_{args.data_seed}.json"

    ret_info = train_fn(args=args, training_data=data_train_file, validation_data=data_dev_file,
                        testing_data=data_test_file)
    train_rse, valid_rse, test_rse = ret_info[-3:]

    if args.use_log:
        wandb.run.summary["final_train_rse"] = train_rse
        wandb.run.summary["final_valid_rse"] = valid_rse
        wandb.run.summary["final_test_rse"] = test_rse

    # Plot simulation distribution
    # data_train = json.load(open(f"{file_name}_train_{args.train_ratio}_{args.seed}.json"))
    # distribution_plot(simulation=[dataum['reward'] for dataum in data_train],
    #                   predictions=[], file_name=f"train_{args.train_ratio}_{args.seed}.jpg", rse=0.0)
