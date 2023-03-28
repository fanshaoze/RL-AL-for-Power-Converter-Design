from gym_envs.rewardUncertainEnv import RewardUncertainEnv
import numpy as np


def vi(env:RewardUncertainEnv, gamma, max_ep_len, pi_info=None):
    states = env.get_discrete_states()
    actions = env.get_discrete_actions()

    q = {(env.hash_state(state), env.hash_action(act)): 0 for state in states for act in actions}
    if pi_info is not None:
        try:
            v = pi_info['v']
        except IndexError:
            raise Exception('v is not in pi_info.')
    else:
        v = {env.hash_state(state): 0 for state in states}

    for step in range(max_ep_len):
        delta = 0
        for s in states:
            if not env.is_terminal(s):
                old_v = v[env.hash_state(s)]
                for a in actions:
                    env.state = s
                    next_s, r, _, _ = env.step(a)
                    q[env.hash_state(s), env.hash_action(a)] = r + gamma * v[env.hash_state(next_s)]
                v[env.hash_state(s)] = max(q[env.hash_state(s), env.hash_action(a)] for a in actions)
                delta = max(delta, abs(v[env.hash_state(s)] - old_v))

        if delta < 1e-2:
            break

    # find the optimal policy
    env.reset()

    step = 0
    traj = [env.state]
    while not env.is_terminal(env.state) and step < max_ep_len:
        opt_act = max(actions, key=lambda a: q[env.hash_state(env.state), env.hash_action(a)])
        env.step(opt_act)
        traj.append(env.state)
        step += 1
    traj = np.array(traj)

    return traj, {'v': v}
