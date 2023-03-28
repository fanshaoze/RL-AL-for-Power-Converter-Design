import random

import numpy as np


# def ucb(means, stds=None, query_time=1):
#     """
#     Logic of ucb selection.
#     Return the id of bandit selected given bandits and the current select time (starting from 1)
#     """
#     # ucb exploration weight
#     #beta = math.sqrt(2 * math.log(size * query_time ** 2 * math.pi ** 2 / (6 * 0.05)))
#     beta = 0.0
#
#     # only consider bandits where std > 0
#     # TODO: is >=0 right?
#     bandits = [mean + beta * std if std > 0 else -np.inf
#                for mean, std in zip(means, stds)]
#     return np.argmax(bandits)
#
# def random_select_state(means, stds=None, query_time=1):
#     """
#     Logic of ucb selection.
#     Return the id of bandit selected given bandits and the current select time (starting from 1)
#     """
#     # ucb exploration weight
#     #beta = math.sqrt(2 * math.log(size * query_time ** 2 * math.pi ** 2 / (6 * 0.05)))
#     beta = 0.0
#
#     # only consider bandits where std > 0
#     bandits = [random.random() if std > 0 else -np.inf
#                for mean, std in zip(means, stds)]
#     return np.argmax(bandits)
def ucb(means, stds=None, query_time=1):
    """
    Logic of ucb selection.
    Return the id of bandit selected given bandits and the current select time (starting from 1)
    """
    # ucb exploration weight
    #beta = math.sqrt(2 * math.log(size * query_time ** 2 * math.pi ** 2 / (6 * 0.05)))
    beta = 0.0

    bandits = [mean + beta * std
               for mean, std in zip(means, stds)]
    return np.argmax(bandits)

def random_select_state(means, stds=None, query_time=1):
    """
    Logic of ucb selection.
    Return the id of bandit selected given bandits and the current select time (starting from 1)
    """
    # ucb exploration weight
    #beta = math.sqrt(2 * math.log(size * query_time ** 2 * math.pi ** 2 / (6 * 0.05)))
    beta = 0.0

    bandits = [random.random()
               for mean, std in zip(means, stds)]
    return np.argmax(bandits)
