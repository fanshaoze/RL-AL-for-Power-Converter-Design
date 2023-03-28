import copy

max_ep_len = 15
gamma = .95
alg = 'vi'

from algs.uct import uct
from algs.vi import vi
from algs.floodfill import floodfill
from spinningup.spinup import ppo_pytorch as ppo


def find_opt_pi(env, seed, plot_file=None, pi_info=None):
    """
    :param env:
    :param seed:
    :param plot_file: plot trajectories and alg-dependent info to files with this prefix
    :return: (optimized traj, return under belief, true return)
    """
    env = copy.deepcopy(env)
    env.reset()

    if pi_info is not None:
        # create a deep copy when available
        pi_info = copy.deepcopy(pi_info)

    if alg == 'ppo':
        traj, info = ppo(
            env=env,
            max_ep_len=max_ep_len, # too short may hinder exploration, the agent never reaches +1
            gamma=gamma,
            ac_kwargs=dict(hidden_sizes=[128, 128]), # [64, 64] seems not working
            epochs=10,
            steps_per_epoch=4000,
            entr_weight=0,
            pi_lr=5e-4,
            vf_lr=1e-4,
            seed=seed,
            plot_file=plot_file)
    elif alg == 'vi':
        traj, info = vi(
            env,
            gamma=gamma,
            max_ep_len=max_ep_len,
            pi_info=pi_info)
    elif alg == 'ff':
        traj, info = floodfill(
            env,
            gamma=gamma,
            max_ep_len=max_ep_len)
    elif alg == 'uct':
        traj, info = uct(
            env,
            max_ep_len=max_ep_len,
            gamma=gamma)
    elif alg == 'uct-old':
        traj, info = uct(
            env,
            max_ep_len=max_ep_len,
            gamma=gamma,
            merge_node=False)
    else:
        raise Exception('unimplemented alg ' + alg)

    if plot_file is not None:
        env.plot_reward_and_policy(file_postfix=plot_file, policy=traj)

    total_r = env.evaluate_traj(traj, gamma)
    true_total_r = env.evaluate_traj(traj, gamma, evaluation=True)

    return traj, total_r, true_total_r, info
