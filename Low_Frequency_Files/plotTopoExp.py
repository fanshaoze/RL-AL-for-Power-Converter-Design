import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_reward(filename, plot_error_bar=True):
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

        if plot_error_bar:
            plt.errorbar(query_nums, reward_means, reward_SEs, marker='+', label='k = ' + k)
        else:
            plt.plot(query_nums, reward_means, '+-', label='k = ' + k)

    plt.xlabel('Number of Queries')
    plt.ylabel('Average Rewards')
    plt.legend()

    plt.savefig(filename + "_reward.png", dpi=300, format="png")
    plt.show()
    plt.close()

def plot_time(filename, plot_error_bar=True):
    results = json.load(open(filename + '.json'))

    for k in results.keys():
        query_nums = []
        time_means = []
        # standard errors
        time_SEs = []

        for traj_num in results[k].keys():
            data = np.array(results[k][traj_num])
            mean = data.mean(axis=0)
            SE = data.std(axis=0) / np.sqrt(data.shape[0])

            time_means.append(mean[3])
            time_SEs.append(SE[3])

            query_nums.append(mean[4])

        print(k, query_nums, time_means)

        if plot_error_bar:
            plt.errorbar(query_nums, time_means, time_SEs, marker='+', label='k = ' + k)
        else:
            plt.plot(query_nums, time_means, '+-', label='k = ' + k)

    plt.xlabel('Number of Queries')
    plt.ylabel('Computation Time (sec.)')
    plt.legend()

    plt.savefig(filename + "_time.png", dpi=300, format="png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, required=True, help='result json file')
    args = parser.parse_args()

    plot_reward(args.data, plot_error_bar=True)
    plot_time(args.data, plot_error_bar=True)
