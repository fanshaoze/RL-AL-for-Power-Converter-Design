import numpy as np


def compute_smooth_reward(eff, vout, target_vout = .5):
    a = abs(target_vout) / 10

    if eff > 1 or eff < 0:
        return 0
    else:
        return eff * (1.1 ** (-((vout - target_vout) / a) ** 2))

def compute_piecewise_linear_reward(eff, vout):
    if eff > 1 or eff < 0 or vout < .35 or vout > .65:
        return 0
    else:
        return eff

compute_reward = compute_smooth_reward

compute_batch_reward = lambda effs, vouts: np.array([compute_reward(e, v) for e, v in zip(effs, vouts)])