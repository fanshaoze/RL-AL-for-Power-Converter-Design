import torch

import numpy as np


def calculate_reward(effi, target_vout=50):
    a = abs(target_vout) / 15
    if effi['efficiency'] > 1 or effi['efficiency'] < 0:
        return 0
    else:
        return effi['efficiency'] * (1.1 ** (-((effi['output_voltage'] - target_vout) / a) ** 2))


def compute_batch_reward(effs, vouts, target_vout):
    assert len(effs) == len(vouts)

    return np.array(
        [calculate_reward({'efficiency': e, 'output_voltage': v}, target_vout) for e, v in zip(effs, vouts)])


def calculate_reward_on_tensor(eff, vout, device, target_vout=0.5):
    """
    Same as above, but eff, vout are tensors
    fixme: here vout is already normalized, between 0 and 1. so target vout should be 0.5
    """
    a = abs(target_vout) / 15

    return torch.where(torch.logical_or(eff > 1, eff < 0),
                       torch.tensor(0.).to(device),  # if the condition above is true, set to 0
                       eff * (1.1 ** (-((vout - target_vout) / a) ** 2)))  # otherwise, compute reward

# debug
# print(calculate_reward_on_tensor(eff=torch.tensor([-1, 2, 1, 0.5, 1]),
#                                  vout=torch.tensor([50, 50, 50]), device='cpu'))
