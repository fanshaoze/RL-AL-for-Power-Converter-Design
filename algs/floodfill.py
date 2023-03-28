from gym_envs.rewardUncertainEnv import RewardUncertainGymEnv
import numpy as np


def floodfill(env:RewardUncertainGymEnv, gamma, max_ep_len):
    """
    Overfit the navigation domain by propagate the positive reward to its neighbors
    """
    states = env.get_discrete_states()
    actions = env.get_discrete_actions()

    v = {}
    discount = 1.

    visited = []
    queue = [env.hash_state(state) for state in states if env.get_reward(state) == 1.]

    backtrack = {}

    while len(queue) > 0:
        for s in queue:
            v[env.hash_state(s)] = 1. * discount

        visited += queue

        neighbors = set()
        for s in queue:
            for a in actions:
                env.state = s
                next_s, r, _, _ = env.step(a)
                next_s_hashable = env.hash_state(next_s)
                # don't add states that have negative rewards, going to detour
                if r == 0. and not next_s_hashable in visited:
                    neighbors.add(next_s_hashable)
                    backtrack[next_s_hashable] = env.hash_state(s)

        queue = list(neighbors)

        discount *= gamma

    # env.plot_values_and_policy(value_func=lambda s: v[env.hash_state(s)] if env.hash_state(s) in v.keys() else 0,
    #                            file_name='state_values')

    env.reset()
    init_s = env.state

    step = 0
    traj = [init_s]
    cur_s = env.hash_state(init_s)

    if cur_s in backtrack.keys():
        # use backtrack to find the shortest path
        while not env.is_terminal(cur_s) and step < max_ep_len:
            cur_s = backtrack[cur_s]
            traj.append(cur_s)

    traj = np.array(traj)

    return traj, {}
