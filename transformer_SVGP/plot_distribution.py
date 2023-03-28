import json
import warnings

import wandb
from matplotlib import pyplot as plt
import numpy as np

def plot_distribution(all_y, all_pred, target, use_wandb=False, plot_file=None):
    bins = 100
    plot_area = np.zeros((bins, bins))
    for y, pred in zip(all_y, all_pred):
        # clip to make sure it's within the plot area
        y_int = np.clip(int(y * bins), 0, bins - 1)
        pred_int = np.clip(int(pred * bins), 0, bins - 1)
        plot_area[pred_int][y_int] += 1

    # plot log scale
    plot_area = np.clip(plot_area, 0, 800)
    plot_area = np.where(plot_area != 0, np.log(plot_area), np.zeros_like(plot_area))

    plt.figure(figsize=(5, 4))
    if target == 'eff':
        axis = [0, 1, 1, 0]
    elif target == 'vout':
        axis = [-3, 3, 3, -3]
    else:
        raise Exception(f"don't know the axis for {target}.")

    plt.imshow(plot_area, extent=axis)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Ground Truth {target}")
    plt.ylabel(f"Surrogate Model Prediction")

    plt.tight_layout()

    if use_wandb:
        if wandb.run is not None:
            data = [[y, pred] for y, pred in zip(all_y, all_pred)]
            data_table = wandb.Table(data=data, columns=["gt", "pred"])
            wandb.log({"chart": plt, "gt_vs_surrogate": data_table})
        else:
            warnings.warn('use_wandb requested, but wandb is not running.')

    if plot_file is not None:
        plt.savefig(f"{plot_file}.png", dpi=300, format="png")

    plt.close()


if __name__ == '__main__':
    filename = "dataset_5_valid_set_lstm_vout_0_perform"

    with open(filename + '.json', 'r') as f:
        all_y, all_pred = json.load(f)
        plot_distribution(all_y, all_pred, 'vout', plot_file=filename)