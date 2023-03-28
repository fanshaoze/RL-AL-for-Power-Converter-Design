import collections
import numpy as np

import wandb

from topo_utils.plot import plot_line

api = wandb.Api()
runs = api.runs("shunzh/surrogate_model")

target = "vout"
results = collections.defaultdict(list)

titles = {'eff': 'Efficiency', 'vout': 'Voltage Output'}

for run in runs:
    if run.name == "dataset_5_cleaned_label"\
            and run.config['target'] == target\
            and run.config['n_layers'] == 2\
            and run.config['duty_encoding'] == 'path'\
            and ('split_by' in run.config and run.config['split_by'] == 'data'):
        key = run.config['train_ratio']

        summary = run.summary._json_dict
        if 'final_test_rse' not in summary:
            continue

        results[key].append(summary['final_test_rse'])

keys = sorted(list(results.keys()))

nums = [len(results[k]) for k in keys]
means = [np.mean(results[k]) for k in keys]
print(nums, means)

plot_line(x_values=keys,
          y_values=means,
          x_label="Training Set Ratio",
          y_label="Relative Squared Error",
          title=titles[target],
          filename=target + '_rse')