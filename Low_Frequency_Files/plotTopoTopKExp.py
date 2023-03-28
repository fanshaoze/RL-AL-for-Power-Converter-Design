import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_line(x, y, label, se=None, plot_error_bar=False):
    if plot_error_bar:
        plt.errorbar(x, y, se, marker='+', ls='-', label=label)
    else:
        plt.plot(x, y, '+-', label=label)

def plot_topo_query_exp(filename, plot_error_bar=True):
    results = json.load(open(filename + '.json'))

    for k in results.keys():
        query_nums = []
        reward_means = []
        # standard errors
        reward_SEs = []

        for traj_num in results[k].keys():
            data = np.array(results[k][traj_num])
            mean = data.mean(axis=0)
            SE = data.std(axis=0) / np.sqrt(data.shape[0])

            reward_means.append(mean[2])
            reward_SEs.append(SE[2])

            query_nums.append(mean[4])

        print(k, query_nums, reward_means)
        plot_line(query_nums, reward_means, label='k = ' + k, se=reward_SEs, plot_error_bar=plot_error_bar)

    plt.xlabel('Number of Queries')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.savefig(filename + ".png", dpi=300, format="png")
    plt.show()
    plt.close()

def modifier(plot_error_bar):
    """
    An ad-hoc function to add extra lines here.
    """
    query_nums = []
    sim_means = []
    sim_stds = []
    sim_ses = []
    plot_line(query_nums, sim_means, label='simulator', se=sim_stds, plot_error_bar=plot_error_bar)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, required=True, help='result json file')
    parser.add_argument('-error-bar', action='store_true')
    args = parser.parse_args()

    plot_topo_query_exp(args.data, plot_error_bar=args.error_bar)