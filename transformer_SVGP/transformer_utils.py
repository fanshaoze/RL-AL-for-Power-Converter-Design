import csv
import json
import os
import sys
import warnings
from types import SimpleNamespace

import gpytorch
import torch

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformer_config
from GetReward import compute_batch_reward
from test_model import test

sys.path.append(os.path.join(sys.path[0], '../'))  # need some functions in uct
from Models import get_model, GPModel

"""
These are helper functions added later useful for Transformer.
"""


def transpose(matrix):
    """

    :param matrix:
    :return:
    """
    return [list(x) for x in zip(*matrix)]


def load_model_from_file(file_name, device):
    """
    load transformer model, using Yupeng's code
    """
    # fixme need to pass target here
    model = get_model(pretrained_model=file_name, load_weights=True)
    model = model.to(device)

    checkpoint = torch.load(file_name + '.chkpt')

    gp_para = checkpoint["gp_model"]
    gp = GPModel(gp_para["variational_strategy.inducing_points"])
    gp.load_state_dict(gp_para)
    gp = gp.to(device)

    likelihood_para = checkpoint['likelihood']
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(likelihood_para)
    likelihood = likelihood.to(device)

    if 'data_train' in checkpoint.keys():
        training_data = checkpoint['data_train']
    else:
        # okay to omit training data
        training_data = None

    model.eval()
    gp.eval()

    return dict(model=model, gp=gp, likelihood=likelihood, data_train=training_data)


def get_transformer_predict(paths, duty, model, gp, vocab, device, use_gp):
    """
    A standalone function that gets transformer prediction for a single data point
    :return: mean, std
    """
    with torch.no_grad(): #, gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        if len(paths) >= transformer_config.max_path_num:
            paths = paths[:transformer_config.max_path_num]

        embedded_paths = []
        for path in paths:
            comps = path.split(' - ')
            embedded_path = [vocab(comp) for comp in comps]
            if len(embedded_path) < transformer_config.max_path_len:
                # make sure all paths have the same length
                # 0 is hardcoded to be <PAD>
                embedded_path += [0] * (transformer_config.max_path_len - len(embedded_path))

            embedded_paths.append(embedded_path)

        # create a dummy dimension for batch size (batch size = 1)
        # since we only have one topology, no need to pad path embeddings
        path_tensor = torch.Tensor(embedded_paths).unsqueeze(0).long()

        # mark component positions in the tensor
        padding_mask = (path_tensor != 0)

        path_tensor = path_tensor.to(device)
        padding_mask = padding_mask.to(device)

        duty = torch.Tensor([duty]).unsqueeze(0).to(device)
        # print('s-path:', path_tensor)
        # print('s-pad:', padding_mask)

        pred, final = model(path_tensor, duty, padding_mask)
        # print('s pred after model', pred)

        if use_gp:
            pred = gp(pred)
            # print('s pred after item', pred.mean.item())
            return pred.mean.item(), np.sqrt(pred.variance.item())
        else:
            # print('s pred after item', final.item())
            # if not using gp, output the final prediction after MLP
            return final.item(), 0


def get_top_k_predictions(surrogate_rewards, ground_truths, k_list=(1, 10, 50, 100)):
    """
    :return: {k: the highest gt reward in the k topologies that have the highest surrogate rewards}
    """
    assert len(surrogate_rewards) == len(ground_truths), 'lengths of surrogate rewards and ground truths are unequal.'

    if len(surrogate_rewards) == 0 or len(ground_truths) == 0:
        warnings.warn('empty state list for top-k evaluation.')

    top_k = {}

    # sort the indices of all topologies by their surrogate rewards DESCENDINGLY
    candidate_indices = np.array(surrogate_rewards).argsort()[::-1]

    for k in k_list:
        # get the corresponding gt rewards of top-k topologies
        true_rewards = [ground_truths[idx] for idx in candidate_indices[:k]]

        # get the highest gt reward of the top-k topologies
        top_k[k] = max(true_rewards)

    return top_k


def evaluate_eff_vout_model(_model, vocab, data, device, use_gp, target, batch_size=512):
    """
    Evaluate an eff model and a vout model on test_data
    :param eff_model: the eff model, or its file location
    :param vout_model: the vout model, or its file location
    :param data: the dataset file location
    :return: {(eff_model_seed, vout_model_seed): top k performance}
    """
    if type(_model) is str:
        test_model = load_model_from_file(_model, device)
    else:
        test_model = dict(model=_model.model, gp=_model.gp)


    # create an argument object for testing
    test_args = SimpleNamespace(data_test=data,
                                vocab=vocab,
                                batch_size=batch_size,
                                max_seq_len=transformer_config.max_path_num,
                                attribute_len=transformer_config.max_path_len,
                                use_gp=use_gp,
                                test_ground_truth='simulation',
                                pretrained_validity_model=None,
                                use_log=False,
                                plot_outliers=False,
                                target=target,
                                device=device)
    print('test_args-----', test_args)

    # rse is returned in the second field, but not needed
    print(test_args.target)
    predicts, _rse = test(test_args, model=test_model['model'], gp=test_model['gp'], testing_data=data)
    if target == 'eff':
        predicts = np.clip(predicts, 0, 1)
    else:
        predicts = np.clip(600. * predicts - 300, -300., 300.)
    return predicts, _rse



def evaluate_model(eff_model, vout_model, vocab, data, device, use_gp, target_vout=50):
    """
    Evaluate an eff model and a vout model on test_data
    :param eff_model: the eff model, or its file location
    :param vout_model: the vout model, or its file location
    :param data: the dataset file location
    :return: {(eff_model_seed, vout_model_seed): top k performance}
    """
    if type(eff_model) is str:
        eff_model = load_model_from_file(eff_model, device)
    if type(vout_model) is str:
        vout_model = load_model_from_file(vout_model, device)

    dataset = json.load(open(data, 'r'))

    # create an argument object for testing
    test_args = SimpleNamespace(data_test=data,
                                vocab=vocab,
                                batch_size=512,
                                max_seq_len=transformer_config.max_path_num,
                                attribute_len=transformer_config.max_path_len,
                                use_gp=use_gp,
                                test_ground_truth='simulation',
                                pretrained_validity_model=None,
                                use_log=False,
                                plot_outliers=False,
                                device=device)

    test_args.target = 'eff'
    # rse is returned in the second field, but not needed
    eff_predicts, _ = test(test_args, model=eff_model['model'], gp=eff_model['gp'], testing_data=data)
    eff_predicts = np.clip(eff_predicts, 0, 1)

    test_args.target = 'vout'
    vout_predicts, _ = test(test_args, model=vout_model['model'], gp=vout_model['gp'], testing_data=data)
    # vout_predicts = np.clip(100 * vout_predicts, 0, 100)
    vout_predicts = np.clip(600. * vout_predicts - 300, -300., 300.)

    surrogate_rewards = compute_batch_reward(eff_predicts, vout_predicts, target_vout=target_vout)
    gt_rewards = [datum['reward'] for datum in dataset]

    return surrogate_rewards, gt_rewards, get_top_k_predictions(surrogate_rewards,
                                                                gt_rewards), eff_predicts, vout_predicts
