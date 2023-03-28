import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_topo_query_exp(filename):
    results = json.load(open(filename + '.json'))

    for iter, result in results.items():
        max_len = max(len(line) for line in result)
        result = list(filter(lambda e: len(e) == max_len, result)) # get rid of incomplete runs

        result = np.array(result)
        avg_result = np.mean(result, axis=0)

        query_nums = []
        rewards = []
        idx = 0
        while idx < len(avg_result):
            query_nums.append(avg_result[idx + 1])
            rewards.append(avg_result[idx + 2])
            idx += 8

        print(query_nums, rewards)
        plt.plot(query_nums, rewards, marker='+', label=iter + ' iter')

    plt.xlabel('Number of Queries')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.title('Active Learning (Retrain and Rebuild Tree)')
    plt.savefig(filename + ".png", dpi=300, format="png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, required=True, help='result json file')
    args = parser.parse_args()

    plot_topo_query_exp(args.data)