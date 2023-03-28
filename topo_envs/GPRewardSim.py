import datetime

import torch

from algs.gp import GPModel
from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim

from topo_data_util.embedding import tf_embed

import numpy as np


def load_gp_model(filename, debug=False):
    data = torch.load(filename)
    train_x = data['train_x']
    train_y = data['train_y']
    state_dict = data['model_state_dict']
    vec_of_paths = data['vec_of_paths']

    gp = GPModel(train_x, train_y, state_dict=state_dict)
    # check_gp(self.reward_model, train_x[:20], train_y[:20])

    if debug:
        print('check gp on training data')
        for idx in range(20):
            print(gp.get_mean(train_x[idx]), train_y[idx])

    return gp, vec_of_paths


class GPRewardTopologySim(SurrogateRewardTopologySim):
    def __init__(self, eff_file, vout_file, debug, *args):
        super().__init__(debug, *args)

        eff_gp, vec_of_paths = load_gp_model(eff_file, debug)
        vout_gp, vec_of_paths = load_gp_model(vout_file, debug)

        # init is called later
        self.eff_gp = eff_gp
        self.vout_gp = vout_gp
        self.vec_of_paths = vec_of_paths

    def get_surrogate_eff(self, state):
        self.set_state(state)

        # convert graph to paths, and find embedding
        paths = self.find_paths()
        embedding = tf_embed(paths, self.vec_of_paths)

        eff = np.clip(self.eff_gp.get_mean(embedding), 0., 1.)

        return eff

    def get_surrogate_vout(self, state):
        self.set_state(state)

        # convert graph to paths, and find embedding
        paths = self.find_paths()
        embedding = tf_embed(paths, self.vec_of_paths)

        vout = np.clip(self.vout_gp.get_mean(embedding), 0., 50.)

        return vout
