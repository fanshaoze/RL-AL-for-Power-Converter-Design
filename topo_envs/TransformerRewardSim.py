import copy
import gc
import json
import math
import random
from dataclasses import dataclass

import gpytorch
import torch
import numpy as np
import statistics

from topoUtils import find_paths, find_paths_with_topo_info
from transformer_SVGP.train import main as train_transformer
from transformer_SVGP.Models import get_model, GPModel
from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim, SurrogateRewardSimFactory
from transformer_SVGP.transformer_utils import get_transformer_predict
from transformer_SVGP.dataset import Dataset
from transformer_SVGP.GetReward import compute_batch_reward

from transformer_SVGP.build_vocab import Vocabulary
from UCT_5_UCB_unblc_restruct_DP_v1.ucts.GetReward import calculate_reward

from topo_data_util.topo_analysis.topoGraph import TopoGraph
from PM_GNN.code.gen_topo_for_dateset import convert_to_netlist
from transformer_SVGP.transformer_utils import evaluate_eff_vout_model
from topoUtils import find_paths, get_topo_key
from transformer_SVGP.evaluate_transformer import transpose, ensemble_information_generation, vote_and_count


@dataclass
class ModelInfo:
    model: ...
    gp: ...
    likelihood: ...
    # data_train: ...


class TransformerRewardSimFactory(SurrogateRewardSimFactory):
    """
    A class that can generate a TransformerRewardSim initializer
    """

    # args used for training the transformer, obtained from Yupeng.
    def __init__(self, eff_model_file, vout_model_file, eff_model_files, vout_model_files, vocab_file,
                 device, training_data, dev_file=None, test_file=None, eff_model_seed=None, vout_model_seed=None,
                 eff_model_seeds=None, vout_model_seeds=None,
                 epoch=10000,
                 patience=50,
                 sample_ratio=0.1):
        """
        Initially, the pretrained models are loaded from files
        """
        if vout_model_seeds is None:
            vout_model_seeds = []
        if eff_model_seeds is None:
            eff_model_seeds = []

        self.vout_model = None
        self.eff_model = None

        self.vout_models = []
        self.eff_models = []

        self.training_data = training_data
        self.AL_simulated_data = Dataset(data_file_name=None, vocab=self.training_data.vocab,
                                         max_seq_len=self.training_data.seq_len,
                                         label_len=self.training_data.label_len)

        self.device = device

        vocab = Vocabulary()
        self.vocab_file = vocab_file
        vocab.load(vocab_file)
        self.vocab = vocab

        self.dev_file = dev_file
        self.test_file = test_file

        self.eff_model_seed = eff_model_seed
        self.vout_model_seed = vout_model_seed
        self.eff_model_file = eff_model_file
        self.vout_model_file = vout_model_file

        self.eff_model_seeds = eff_model_seeds
        self.vout_model_seeds = vout_model_seeds
        self.eff_model_files = eff_model_files
        self.vout_model_files = vout_model_files

        self.reset_model()

        self.epoch = epoch
        self.patience = patience
        self.sample_ratio = sample_ratio

        self.surrogate_hash_table = {}
        self.ensemble_surrogate_hash_table = {}
        self.eff_var_hash_table = {}
        self.vout_var_hash_table = {}

    def reset_surrogate_table_with_graph_to_reward(self, graph_to_reward, target_vout):
        """

        @return:
        """
        self.surrogate_hash_table, self.eff_var_hash_table, self.vout_var_hash_table = {}, {}, {}
        for k, v in graph_to_reward.items():
            eff_obj = {'efficiency': float(v[1]), 'output_voltage': float(v[2])}
            self.surrogate_hash_table[k] = calculate_reward(eff_obj, target_vout)
            self.eff_var_hash_table[k] = 0
            self.vout_var_hash_table[k] = 0

    def reset_model(self):
        # loaded in (model, gp, likelihood) tuples
        self.eff_model = self.load_model_from_file(self.eff_model_file)
        self.vout_model = self.load_model_from_file(self.vout_model_file)
        self.eff_models = [self.load_model_from_file(eff_model_file) for eff_model_file in self.eff_model_files]
        self.vout_models = [self.load_model_from_file(vout_model_file) for vout_model_file in self.vout_model_files]
        self.AL_simulated_data = Dataset(data_file_name=None, vocab=self.training_data.vocab,
                                         max_seq_len=self.training_data.seq_len,
                                         label_len=self.training_data.label_len)

    def set_models(self, tmp_eff_model, tmp_vout_model, tmp_eff_models, tmp_vout_models):
        # loaded in (model, gp, likelihood) tuples
        del self.eff_model
        del self.vout_model
        del self.eff_models
        del self.vout_models
        self.eff_model = copy.deepcopy(tmp_eff_model)
        self.vout_model = copy.deepcopy(tmp_vout_model)
        self.eff_models = copy.deepcopy(tmp_eff_models)
        self.vout_models = copy.deepcopy(tmp_vout_models)
        # gc.collect()

    def load_model_from_file(self, file_name):
        # load transformer model, using Yupeng's code
        # model = get_model(cuda=(self.device == 'gpu'), pretrained_model=file_name, load_weights=True)
        model = get_model(pretrained_model=file_name, load_weights=True)
        model = model.to(self.device)

        checkpoint = torch.load(file_name + '.chkpt')

        gp_para = checkpoint["gp_model"]
        gp = GPModel(gp_para["variational_strategy.inducing_points"])
        gp.load_state_dict(gp_para)
        gp = gp.to(self.device)

        likelihood_para = checkpoint['likelihood']
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.load_state_dict(likelihood_para)
        likelihood = likelihood.to(self.device)

        if 'data_train' in checkpoint.keys():
            training_data = checkpoint['data_train']
        else:
            # okay to omit training data
            training_data = None

        model.eval()
        gp.eval()

        # return ModelInfo(model, gp, likelihood, training_data)
        return ModelInfo(model, gp, likelihood)

    # def add_data_to_model_and_train(self, path_set, effs, vouts):
    def add_data_to_model_and_train(self, path_set, duties, effs, vouts, valids, rewards):
        # init training data for this AL
        training_data = Dataset(data_file_name=None, vocab=self.training_data.vocab,
                                max_seq_len=self.training_data.seq_len, label_len=self.training_data.label_len)
        # add new simulated data, sampled old data, sampled pre simulated data and
        # update AL simulated data with newly simulated data
        training_data.append_data(path_set, duties, effs, vouts, valids, rewards)
        data, idx = self.training_data.random_sample_data(ratio=self.sample_ratio)
        training_data.merge_data_with_data_idx(_data=data, idx=idx)
        data, idx = self.AL_simulated_data.random_sample_data(ratio=self.sample_ratio)
        training_data.merge_data_with_data_idx(_data=data, idx=idx)
        self.AL_simulated_data.append_data(path_set, duties, effs, vouts, valids, rewards)

        # training_data = self.eff_models.data_train
        # print(len(training_data))
        # pre_training_data = copy.deepcopy(self.eff_models.data_train)
        # assert training_data is not None
        # # TODO sample must include all good(rank and sample)
        # # TODO remove the duplicated
        # training_data.random_sample_data(ratio=self.sample_ratio)
        # # training_data.get_trained_data_with_fix_good_topo(get_length=math.ceil(
        # #                                                              len(training_data) * self.sample_ratio))
        #
        # # training_data.append_data(path_set, effs, vouts)
        # # pre_training_data.append_data(path_set, effs, vouts)
        # training_data.append_data(path_set, duties, effs, vouts, valids)
        # pre_training_data.append_data(path_set, duties, effs, vouts, valids)

        # keep training the transformer, while using the current trained models
        new_eff_model = train_transformer(args=['-data_dev=' + self.dev_file,
                                                '-data_test=' + '',
                                                # '-data_test=' + self.test_file,
                                                '-target=eff',
                                                '-vocab=' + self.vocab_file,
                                                '-seed=' + str(self.eff_model_seed)],
                                          training_data=training_data,
                                          transformer=self.eff_model.model,
                                          gp=self.eff_model.gp,
                                          likelihood=self.eff_model.likelihood,
                                          epoch=self.epoch,
                                          patience=self.patience)

        # self.eff_models = ModelInfo(*(new_eff_models[0], new_eff_models[1],
        #                               new_eff_models[2], new_eff_models[3]))
        self.eff_model = ModelInfo(*(new_eff_model[0], new_eff_model[1],
                                     new_eff_model[2]))
        effi_early_stop = new_eff_model[4]
        epoch_i_eff = new_eff_model[5]

        new_vout_model = train_transformer(args=['-data_dev=' + self.dev_file,
                                                 '-data_test=' + '',
                                                 # '-data_test=' + self.test_file,
                                                 '-target=vout',
                                                 '-vocab=' + self.vocab_file,
                                                 '-seed=' + str(self.vout_model_seed)],
                                           training_data=training_data,
                                           transformer=self.vout_model.model,
                                           gp=self.vout_model.gp,
                                           likelihood=self.vout_model.likelihood,
                                           epoch=self.epoch,
                                           patience=self.patience)
        # self.vout_models = ModelInfo(*(new_vout_models[0], new_vout_models[1],
        #                                new_vout_models[2], new_vout_models[3]))
        self.vout_model = ModelInfo(*(new_vout_model[0], new_vout_model[1],
                                      new_vout_model[2]))
        vout_early_stop = new_vout_model[4]
        epoch_i_vout = new_vout_model[5]

        # self.eff_models.data_train = pre_training_data
        # print(self.eff_models)
        return effi_early_stop, vout_early_stop, epoch_i_eff, epoch_i_vout

    def update_sim_models(self, sim):
        sim.eff_model = self.eff_model
        sim.vout_model = self.vout_model
        sim.eff_models = self.eff_models
        sim.vout_models = self.vout_models

    def get_sim_init(self):
        return lambda *args: TransformerRewardSim(
            self.eff_model, self.vout_model, self.eff_models, self.vout_models, self.vocab, self.device,
            self.surrogate_hash_table, self.ensemble_surrogate_hash_table, self.eff_var_hash_table,
            self.vout_var_hash_table, *args)


def generate_range_list(range_bounds):
    range_list = []
    for i in range(len(range_bounds) - 1):
        range_list.append((range_bounds[i], range_bounds[i + 1]))
    return range_list


def get_raw_data(state_list):
    raw_data = []
    for _state in state_list:
        reward = 0
        name = get_topo_key(_state) + '$' + str(_state.parameters)
        list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(_state.graph, _state.component_pool,
                                                                             _state.port_pool,
                                                                             _state.parent, _state.comp2port_mapping)
        paths = TopoGraph(node_list=list_of_node,
                          edge_list=list_of_edge).find_end_points_paths_as_str()
        datum = {
            "name": name,
            "list_of_edges": list_of_edge,
            "list_of_nodes": list_of_node,
            "paths": paths,
            "eff": 0.5,
            "eff_analytic": 0.,
            "vout": 50,
            "vout_analytic": 0.,
            "duty": _state.parameters[0],
            "valid": True,
            "reward": 0
        }
        raw_data.append(datum)
    return raw_data


class TransformerRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model, vout_model, eff_models, vout_models, vocab, device, _surrogate_hash_table,
                 _ensemble_surrogate_hash_table, _eff_var_hash_table, _vout_var_hash_table, *args):
        super().__init__(_surrogate_hash_table, _ensemble_surrogate_hash_table, _eff_var_hash_table,
                         _vout_var_hash_table, *args)

        self.eff_model = eff_model
        self.vout_model = vout_model
        self.eff_models = eff_models
        self.vout_models = vout_models
        self.eff_range_bounds = [0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + math.pow(0.1, 10)]
        self.vout_range_bounds = [-300, -200, -100, -50, 0, 25, 45, 55, 75, 90, 110, 200, 300 + math.pow(0.1, 10)]
        self.eff_range_list = generate_range_list(self.eff_range_bounds)
        self.vout_range_list = generate_range_list(self.vout_range_bounds)

        self.vocab = vocab

        self.device = device

    def get_transformer_predict(self, state, model, gp, need_std=False):
        """

        @param state:
        @param model:
        @param gp:
        @param need_std:
        @return:
        """
        paths = find_paths(state)
        duty = state.parameters[0]
        # mean, std = get_transformer_predict(paths, duty, model, gp, self.vocab, self.device, use_gp=True)
        mean, std = get_transformer_predict(paths, duty, model, gp, self.vocab, self.device, use_gp=True)

        if need_std:
            return std
        else:
            return mean

    def get_transformer_predict_with_topo_info(self, node_list, edge_list, duty, model, gp, need_std=False):
        """
        get the transformer prediction with topology information
        @param node_list:
        @param edge_list:
        @param duty:
        @param model:
        @param gp:
        @param need_std:
        @return:
        """
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            paths = find_paths_with_topo_info(node_list=node_list, edge_list=edge_list)
            mean, std = get_transformer_predict(paths, duty, model, gp, self.vocab, self.device, use_gp=True)

        if need_std:
            return std
        else:
            return mean

    def get_surrogate_eff(self, state):
        """
        get surrogate model vout, needs de-normalizing the vout
        @param state:
        @return:
        """
        return np.clip(self.get_transformer_predict(state, self.eff_model.model, self.eff_model.gp), 0., 1.)

    def get_surrogate_vout(self, state):
        """
        get surrogate model vout, needs de-normalizing the vout from 0, 1 to -300, +300
        @param state:
        @return:
        """
        return np.clip(600. * self.get_transformer_predict(state, self.vout_model.model, self.vout_model.gp) - 300,
                       -300., 300.)

    def get_surrogate_eff_with_topo_info(self, node_list, edge_list, duty):
        """
        get surrogate model efficiency using node_list and edge_list
        @param node_list:
        @param edge_list:
        @param duty:
        @return:
        """
        return np.clip(self.get_transformer_predict_with_topo_info(duty=duty, node_list=node_list, edge_list=edge_list,
                                                                   model=self.eff_model.model,
                                                                   gp=self.eff_model.gp), 0., 1.)

    def get_surrogate_vout_with_topo_info(self, node_list, edge_list, duty):
        """
        get surrogate model output voltage using node_list and edge_list
        @param node_list:
        @param edge_list:
        @param duty:
        @return:
        """
        return np.clip(
            600. * self.get_transformer_predict_with_topo_info(duty=duty, node_list=node_list, edge_list=edge_list,
                                                               model=self.vout_model.model,
                                                               gp=self.vout_model.gp) - 300., -300., 300.)

    def get_surrogate_reward(self, state):
        """
        fixme this should be implemented in surrogateRewardSim.py, haven't merged it yet
        """
        eff = self.get_surrogate_eff(state)
        vout = self.get_surrogate_vout(state)

        eff_obj = {'efficiency': float(eff), 'output_voltage': float(vout)}
        if self.configs_['round'] == 'vout':
            reward = float(eff) * float(vout) / 100
        else:
            reward = calculate_reward(eff_obj, self.configs_['target_vout'])

        return eff, vout, reward, state.parameters

    def get_surrogate_reward_with_topo_info(self, node_list, edge_list, duty):
        """
        fixme this should be implemented in surrogateRewardSim.py, haven't merged it yet
        """
        eff = self.get_surrogate_eff_with_topo_info(node_list, edge_list, duty)
        vout = self.get_surrogate_vout_with_topo_info(node_list, edge_list, duty)

        eff_obj = {'efficiency': float(eff), 'output_voltage': float(vout)}
        if self.configs_['round'] == 'vout':
            reward = float(eff) * float(vout) / 100
        else:
            reward = calculate_reward(eff_obj, self.configs_['target_vout'])

        return eff, vout, reward

    def get_ensemble_surrogate_eff(self, state):
        """
        get surrogate model vout, needs de-normalizing the vout
        @param state:
        @return:
        """
        # TODO: ensemble
        predictions = []
        for idx, eff_model in enumerate(self.eff_models):
            predictions.append(np.clip(self.get_transformer_predict(state, eff_model.model, eff_model.gp), 0., 1.))
        print(predictions)
        vote_hash, voted_range = vote_and_count(values=predictions, range_list=self.eff_range_list)
        if len(self.eff_models) > 1:
            return statistics.mean([float(predictions[idx]) for idx in vote_hash[voted_range]]), \
                   statistics.stdev([float(predictions[idx]) for idx in range(len(predictions))])
        else:
            return statistics.mean([float(predictions[idx]) for idx in vote_hash[voted_range]]), 0

    def get_ensemble_surrogate_vout(self, state):
        """
        get surrogate model vout, needs de-normalizing the vout from 0, 1 to -300, +300
        @param state:
        @return:
        """
        # TODO: ensemble
        predictions = []
        for idx, vout_model in enumerate(self.vout_models):
            predictions.append(
                np.clip(600. * self.get_transformer_predict(state, vout_model.model, vout_model.gp) - 300,
                        -300., 300.))
        print('vout:------', predictions)
        vote_hash, voted_range = vote_and_count(values=predictions, range_list=self.vout_range_list)
        if len(self.vout_models) > 1:
            return statistics.mean([float(predictions[idx]) for idx in vote_hash[voted_range]]), \
                   statistics.stdev([float(predictions[idx]) for idx in range(len(predictions))])
        else:
            return statistics.mean([float(predictions[idx]) for idx in vote_hash[voted_range]]), 0

    def get_ensemble_surrogate_eff_with_topo_info(self, node_list, edge_list, duty):
        """
        get surrogate model efficiency using node_list and edge_list
        @param node_list:
        @param edge_list:
        @param duty:
        @return:
        """

        predictions = []
        for idx, eff_model in enumerate(self.eff_models):
            predictions.append(
                np.clip(self.get_transformer_predict_with_topo_info(duty=duty, node_list=node_list, edge_list=edge_list,
                                                                    model=eff_model.model,
                                                                    gp=eff_model.gp), 0., 1.))
        vote_hash, voted_range = vote_and_count(values=predictions, range_list=self.eff_range_list)
        return statistics.mean([float(predictions[idx]) for idx in vote_hash[voted_range]]), \
               statistics.stdev([float(predictions[idx]) for idx in range(len(predictions))])

    def get_ensemble_surrogate_vout_with_topo_info(self, node_list, edge_list, duty):
        """
        get surrogate model output voltage using node_list and edge_list
        @param node_list:
        @param edge_list:
        @param duty:
        @return:
        """
        predictions = []
        for idx, vout_model in enumerate(self.vout_models):
            predictions.append(
                np.clip(
                    600. * self.get_transformer_predict_with_topo_info(duty=duty, node_list=node_list,
                                                                       edge_list=edge_list,
                                                                       model=vout_model.model,
                                                                       gp=vout_model.gp) - 300., -300., 300.))
        vote_hash, voted_range = vote_and_count(values=predictions, range_list=self.vout_range_list)
        return statistics.mean([float(predictions[idx]) for idx in vote_hash[voted_range]]), \
               statistics.stdev([float(predictions[idx]) for idx in range(len(predictions))])

    def get_ensemble_surrogate_reward(self, state):
        """
        fixme this should be implemented in surrogateRewardSim.py, haven't merged it yet
        """
        # TODO: ensemble
        eff, eff_std = self.get_ensemble_surrogate_eff(state)
        vout, vout_std = self.get_ensemble_surrogate_vout(state)

        eff_obj = {'efficiency': float(eff), 'output_voltage': float(vout)}
        if self.configs_['round'] == 'vout':
            reward = float(eff) * float(vout) / 100
        else:
            reward = calculate_reward(eff_obj, self.configs_['target_vout'])

        return eff, vout, reward, state.parameters, eff_std, vout_std

    def get_ensemble_surrogate_reward_with_topo_info(self, node_list, edge_list, duty):
        """
        fixme this should be implemented in surrogateRewardSim.py, haven't merged it yet
        """
        eff, eff_std = self.get_ensemble_surrogate_eff_with_topo_info(node_list, edge_list, duty)
        vout, vout_std = self.get_ensemble_surrogate_vout_with_topo_info(node_list, edge_list, duty)

        eff_obj = {'efficiency': float(eff), 'output_voltage': float(vout)}
        if self.configs_['round'] == 'vout':
            reward = float(eff) * float(vout) / 100
        else:
            reward = calculate_reward(eff_obj, self.configs_['target_vout'])

        return eff, vout, reward

    def get_ensemble_surrogate_eff_std(self, state):
        eff_stds = []
        for idx, eff_model in enumerate(self.eff_models):
            eff_stds.append(self.get_transformer_predict(state, eff_model.model, eff_model.gp, need_std=True))
        return statistics.mean(eff_stds)

    def get_ensemble_surrogate_vout_std(self, state):
        vout_stds = []
        for idx, vout_model in enumerate(self.vout_models):
            vout_stds.append(600. * self.get_transformer_predict(state, vout_model.model, vout_model.gp,
                                                                 need_std=True) - 300.)
        return statistics.mean(vout_stds)

    def sequential_generate_ensemble_infos(self, cand_states):
        reward_ensemble_predictions, eff_ensemble_predictions, eff_uncertainty_stds, \
        vout_ensemble_predictions, vout_uncertainty_stds = [], [], [], [], []
        for _state in cand_states:
            name = get_topo_key(_state) + '$' + str(_state.parameters)
            if name in self.ensemble_surrogate_hash_table:
                eff, vout, reward, _, eff_std, vout_std = self.ensemble_surrogate_hash_table[name]
            else:
                eff, vout, reward, paras, eff_std, vout_std = self.get_ensemble_surrogate_reward(_state)
                self.ensemble_surrogate_hash_table[name] = [eff, vout, reward, paras, eff_std, vout_std]
            reward_ensemble_predictions.append(reward)
            eff_ensemble_predictions.append(eff)
            eff_uncertainty_stds.append(eff_std)
            vout_ensemble_predictions.append(vout)
            vout_uncertainty_stds.append(vout_std)
        return reward_ensemble_predictions, eff_ensemble_predictions, eff_uncertainty_stds, \
               vout_ensemble_predictions, vout_uncertainty_stds

    def batch_generate_ensemble_infos(self, cand_states):
        raw_data = get_raw_data(cand_states)
        raw_data_file = 'raw_data_for_retrain.json'
        json.dump(raw_data, open(raw_data_file, 'w'))
        data_pred_effs = []
        data_pred_vouts = []
        for eff_model in self.eff_models:
            eff_preds, _ = evaluate_eff_vout_model(_model=eff_model, vocab=self.vocab,
                                                   data=raw_data_file,
                                                   device=self.device, use_gp=True, target='eff',
                                                   batch_size=4)
            data_pred_effs.append(eff_preds)
        print(data_pred_effs)
        data_pred_effs = transpose(data_pred_effs)

        eff_ensemble_predictions, eff_vote_counts, eff_uncertainty_counts, eff_uncertainty_stds = \
            ensemble_information_generation(data_preds=data_pred_effs, range_bound_list=self.eff_range_bounds)

        for vout_model in self.vout_models:
            vout_preds, _ = evaluate_eff_vout_model(_model=vout_model, vocab=self.vocab,
                                                    data=raw_data_file,
                                                    device=self.device, use_gp=True, target='vout',
                                                    batch_size=4)
            data_pred_vouts.append(vout_preds)
        data_pred_vouts = transpose(data_pred_vouts)
        # print(data_pred_vouts)
        vout_ensemble_predictions, vout_vote_counts, vout_uncertainty_counts, vout_uncertainty_stds = \
            ensemble_information_generation(data_preds=data_pred_vouts, range_bound_list=self.vout_range_bounds)

        reward_ensemble_predictions = compute_batch_reward(eff_ensemble_predictions, vout_ensemble_predictions,
                                                           target_vout=self.configs_['target_vout'])
        return reward_ensemble_predictions, eff_ensemble_predictions, eff_uncertainty_stds, \
               vout_ensemble_predictions, vout_uncertainty_stds
