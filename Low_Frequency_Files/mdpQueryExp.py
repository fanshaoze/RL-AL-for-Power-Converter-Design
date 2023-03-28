import copy
import logging
import time

import gym
import torch

import algs
from algs import policyOpt
from arguments import get_args

from algs.mdpQueryAgents import EPUQueryAgent, UCBQueryAgent, MeanRewardQueryAgent, MaxUncertainQueryAgent
from al_util import feed_random_seeds


def experiment_instance(args, seed):
    feed_random_seeds(seed)

    if args.env == 'Navigation-Goal-Trap':
        from gym_envs.navigation import navigation_2d_goal_and_trap
        env = navigation_2d_goal_and_trap()
    elif args.env == 'Navigation-Random':
        from gym_envs.navigation import navigation_2d_random_targets
        env = navigation_2d_random_targets()
    else:
        # a regular domain
        env = gym.make(args.env)

    env.set_seed(seed)

    # use random or pre-specified training and test sets
    training_set = None #[(.3, .3), (.6, .5)]
    test_set = None #[(.3, .301), (.6, .501), (.6, .6)]

    env.make_reward_model(
        training_size=args.train_size,
        valid_size=args.valid_size,
        sigma=args.sigma,
        data_x=training_set)

    test_x, _ = env.sample_state_reward_pairs(
        args.test_size * args.query_times,
        interesting_points=False,
        data_x=test_set)

    # plot for debugging
    env.plot_reward_variance(file_postfix=str(seed))
    env.plot_eval_reward(file_postfix=str(seed))

    # query candidates
    QueryAgents = [EPUQueryAgent, UCBQueryAgent, MeanRewardQueryAgent, MaxUncertainQueryAgent]

    # store env output stats
    posterior_values = []
    times = []

    # prior optimal pi and value
    prior_opt_pi, prior_value, prior_true_value, pi_info = policyOpt.find_opt_pi(env, seed, plot_file=str(seed))
    logging.info("prior true value " + str(prior_true_value))

    for QueryAgent in QueryAgents:
        agent = QueryAgent(copy.deepcopy(env), seed, pi_info=pi_info)

        # agent specific configs
        if agent.name == 'EPU' and args.epu_responses is not None:
            agent.simulated_responses = args.epu_responses

        logging.info("running " + agent.name)

        start_time = time.time()

        for query_time in range(args.query_times):
            logging.info('query time: ' + str(query_time))

            this_test_x = test_x[query_time * args.test_size : (query_time + 1) * args.test_size]
            agent.set_query_cand(this_test_x)
            logging.info('test x ' + str(this_test_x))

            # consider reward query?
            query = agent.find_query(query_time=query_time + 1)
            true_reward = env.get_true_reward(query)

            logging.info('query is ' + str(query) + ' and true reward is ' + str(true_reward))

            # create the posterior model, need to match the gp data types
            agent.env.update_reward_model(query, true_reward)
            agent.env.queried_data_size += 1

            # find the posterior optimal policy
            poster_opt_policy, _, poster_true_value, _ = policyOpt.find_opt_pi(agent.env, seed, plot_file=agent.name + "_" + str(seed) + "_" + str(query_time), pi_info=pi_info)
            logging.info('posterior pi ' + str(poster_true_value))
            posterior_values.append(poster_true_value)

            # also plot the variance for debugging
            agent.env.plot_reward_variance(file_postfix=agent.name + "_" + str(seed) + "_" + str(query_time))

        end_time = time.time()
        time_elapsed = end_time - start_time

        #posterior_values.append(true_func(post_x))
        times.append(time_elapsed)

    evois = [v - prior_true_value for v in posterior_values]

    logging.info("posterior values " + str(posterior_values))
    logging.info("computation time " + str(times))

    results = {"times": times, "posterior_values": posterior_values, "evois": evois}

    return results


def experiments(args):
    logging.basicConfig(
        level=logging.INFO,
        filename='log',
        filemode='w')
    # print to stderr as well
    logging.getLogger().addHandler(logging.StreamHandler())

    stat_names = ["posterior_values", "evois", "times"]

    time_stamp = time.ctime(time.time())

    # initialize empty tensors
    data = {stat: torch.Tensor([]) for stat in stat_names}

    # set random seed or random seed range
    if args.seed_range is not None:
        seed_range = range(args.seed_range[0], args.seed_range[1])
    elif args.seed is not None:
        seed_range = [args.seed]
    else:
        # just run random seed 0 by default
        seed_range = [0]

    if args.alg is not None:
        policyOpt.alg = args.alg

    if args.num_runs is not None:
        algs.uct.num_runs = args.num_runs

        if not args.alg.startswith('uct'):
            Warning('used num_runs but not using UCT.')

    if args.test:
        # smoke test
        args.test_size = 1
        args.epu_responses = 1

    # run exps for all random seeds
    for seed in seed_range:
        logging.info("random seed %s" % seed)

        exp_data = experiment_instance(args, seed)

        for stat in stat_names:
            this_instance = torch.Tensor(exp_data[stat]).unsqueeze(0)
            data[stat] = torch.cat((data[stat], this_instance))

            if stat in stat_names:
                print_stats(stat, data[stat])

        if not args.dry:
            torch.save(data, "query_selection_" + str(time_stamp) + ".pt")

def print_stats(stat_name, d):
    logging.info(stat_name)
    logging.info('mean')
    logging.info(torch.mean(d, dim=0).numpy())
    logging.info('std')
    logging.info(torch.std(d, dim=0).numpy())

if __name__ == '__main__':
    args = get_args()
    experiments(args)
