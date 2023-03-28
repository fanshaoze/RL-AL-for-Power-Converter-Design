import copy
import logging
import math
import random

import torch

from abc import ABC, abstractmethod

from algs.activeLearning import ucb
from algs.policyOpt import find_opt_pi
from gym_envs.rewardUncertainEnv import RewardUncertainEnv

import numpy as np


class QueryAgent(ABC):
    def __init__(self, env: RewardUncertainEnv, seed: int, pi_info=None):
        """
        :param env:
        :param test_x: The set of possible queries for selection
        :param pi_info: Information needed for optimizing a policy, could be info used for computing the prior optimal pi.
        """
        self.env = env
        self.seed = seed
        self.pi_info = pi_info

    def set_query_cand(self, test_x: torch.Tensor):
        """
        Set the set of data points that the agent may consider as queries
        """
        self.test_x = test_x

    @abstractmethod
    def find_query(self, query_time=None):
        pass


class EPUQueryAgent(QueryAgent):
    name = "EPU"

    def __init__(self, env: RewardUncertainEnv, seed: int,
                 simulated_responses=5,
                 random_sample_response=False,
                 bias='alter',
                 pi_info=None):
        super().__init__(env, seed, pi_info)

        # default simulated response value
        self.simulated_responses = simulated_responses
        self.random_sample_response = random_sample_response
        self.bias = bias

    def find_query(self, query_time=None):
        """
        An exhaustive way to find the best query by computing the EPU of all x
        """
        with torch.no_grad():
            q_values = []

            for query in self.test_x:
                logging.info("query is " + str(query))

                query = torch.Tensor([query])
                pred = self.env.reward_model.posterior(query)
                logging.debug("response stats " + str(pred.mean) + " " + str(pred.variance))

                if self.random_sample_response:
                    responses = pred.sample(sample_shape=torch.Size((self.simulated_responses,)))
                    weights = torch.ones(self.simulated_responses)
                else:
                    post_mean = pred.mean
                    post_std = torch.sqrt(pred.variance)

                    # use weighted sampling. posterior values are weighted by the likelihood
                    # weights re not normalized
                    responses = [post_mean - post_std * 2,
                                 post_mean - post_std,
                                 post_mean,
                                 post_mean + post_std,
                                 post_mean + post_std * 2]
                    if self.bias == None:
                        weights = [0.053990, 0.241970, 0.398942, 0.241970, 0.053990]
                    elif self.bias == 'alter':
                        if query_time % 2 == 1:
                            weights = [0, 0, 0, 0, 1]
                        else:
                            weights = [1, 0, 0, 0, 0]
                    else:
                        raise Exception('unknown bias config in epu: ' + str(self.bias))

                post_values = []
                for idx, response in enumerate(responses):
                    logging.info("sampled response " + str(response))
                    if weights[idx] == 0:
                        # no need to compute it if no weight associated
                        post_value = 0
                        logging.info("skipped")
                    else:
                        # create a copy to compute the posterior reward model
                        post_env = copy.deepcopy(self.env)
                        # need to make a call before posterior update, don't know why
                        post_env.reward_model(query)

                        # fixme does response here work?
                        post_env.update_reward_model(query, response)

                        post_env.queried_data_size = 1 # to mark the simulated query
                        #plot_file = str(query) + '_' + str(idx) + '_' + str(self.seed)
                        plot_file = None
                        _, post_value, _, _ = find_opt_pi(post_env, self.seed, plot_file=plot_file, pi_info=self.pi_info)
                        logging.info("opt post value is " + str(post_value))

                    post_values.append(post_value)

                q_value = np.inner(post_values, weights)

                logging.info('un-normalized epu is ' + str(q_value))
                q_values.append(q_value)

            #bar_plot(self.test_x, q_values, "tau", "EPU", "EPU_values")

            query_and_values = max(zip(self.test_x, q_values), key=lambda _: _[1])
            return query_and_values[0]


class MeanRewardQueryAgent(QueryAgent):
    name = "Mean"

    def __init__(self, env: RewardUncertainEnv, seed: int,
                 bias='alter',
                 pi_info=None):
        super().__init__(env, seed, pi_info)
        self.bias = bias

    def find_query(self, query_time=None):
        with torch.no_grad():
            if self.bias == 'opt':
                return max(self.test_x, key=lambda x: self.env.get_reward_mean(x))
            elif self.bias == 'pas':
                return min(self.test_x, key=lambda x: self.env.get_reward_mean(x))
            elif self.bias == 'alter':
                if query_time % 2 == 1:
                    return max(self.test_x, key=lambda x: self.env.get_reward_mean(x))
                else:
                    return min(self.test_x, key=lambda x: self.env.get_reward_mean(x))
            else:
                raise Exception('unknown bias config in mean reward q agent: ' + str(self.bias))


class MaxUncertainQueryAgent(QueryAgent):
    name = "Uncertain"

    def find_query(self, query_time=None):
        with torch.no_grad():
            return max(self.test_x, key=lambda x: self.env.get_reward_variance(x))


class PIQueryAgent(QueryAgent):
    name = "Probability of Improvement"

    def find_query(self, query_time=None):
        pass


class UCBQueryAgent(QueryAgent):
    name = "UCB"

    def find_query(self, query_time=None):
        assert query_time is not None, 'UCB needs query time input.'
        with torch.no_grad():
            env = self.env

            means = [env.get_reward_mean(x) for x in self.test_x]
            variances = [env.get_reward_variance(x) for x in self.test_x]

            return ucb(means, variances, query_time)


class RandomQueryAgent(QueryAgent):
    name = "Random"

    def find_query(self, query_time=None):
        with torch.no_grad():
            query_idx = random.randint(0, len(self.test_x) - 1)

            return self.test_x[query_idx]
