import torch

from GetReward import calculate_reward_on_tensor


def get_pred_and_y(eff, vout, valid, pred, target, device):
    """
    process ground truth data (eff, vout) and prediction (pred) to compute the training/evaluation loss

    :return: prediction (pred) and ground truth data (y)
    """
    eff = eff.squeeze(1)
    vout = vout.squeeze(1)
    valid = valid.squeeze(1)

    eff = torch.clamp(eff, 0., 1.)
    # [0, 100] -> [0, 1]
    #vout = torch.clamp(vout / 100., 0., 1.)
    # [-300, 300] -> [0, 600] -> [0, 1]
    vout = torch.clamp((vout + 300.) / 600., 0., 1.)

    if target == 'eff':
        # if invalid, set to 0
        y = torch.where(valid > 0.0, eff, torch.zeros_like(eff))
    elif target == 'vout':
        # if invalid, set to 0
        y = torch.where(valid > 0.0, vout, torch.zeros_like(vout))
    elif target == 'eff_vout':
        # [[eff, vout], [eff, vout], ...]
        y = torch.stack((eff, vout), dim=1)

        # convert to [eff * vout, eff * vout, ...
        pred = pred[:, 0] * pred[:, 1]
        y = y[:, 0] * y[:, 1]
    elif target == 'reward':
        # [reward, reward, ...]
        y = calculate_reward_on_tensor(eff, vout, device=device)
    elif target == 'valid':
        y = valid
    else:
        raise Exception('unknown target ' + target)

    return pred, y
