import copy
import logging

import numpy as np

from gym_envs.simulatorWrapper import SimulatorWrapper, GymState, GymAction
from UCF.topo_gen.uct import UCTPlanner

max_depth = 20
num_runs = 1000

def uct(env, max_ep_len=1000, gamma=.99, merge_node=True):
    num_acts = env.action_space.n

    uct_env = copy.deepcopy(env)

    sim = SimulatorWrapper(env)
    uct_simulator = SimulatorWrapper(uct_env)
    uctTree = UCTPlanner(_sim=uct_simulator, _maxDepth=max_depth, _numRuns=num_runs, _ucbScalar=20., _gamma=gamma,
                         _leafValue=0, _endEpisodeValue=0, maxEpLen=max_ep_len, mergeNode=merge_node)
    r = 0
    step = 0

    traj = [env.state]

    uctTree.setRootNode(sim.getState(), sim.getActions(), r, sim.isTerminal())

    def get_q_value(s):
        values = [uctTree.get_return(GymState(s), GymAction(act)) for act in range(num_acts)]
        values = list(filter(lambda _: _ is not None, values))
        if len(values) == 0:
            return 0
        else:
            return max(values)

    while not sim.isTerminal() and step < max_ep_len:
        uctTree.plan()

        # get_visits is a function of a GymState, so create a dummy GymState just to get its visit
        env.plot_values_and_policy(value_func=lambda s: uctTree.get_visits(GymState(s)),
                                   file_name='state_visits')
        env.plot_values_and_policy(value_func=get_q_value,
                                   file_name='state_values')

        # return the action with the highest reward
        action = uctTree.getAction()
        sim.act(action)

        uctTree.updateRootNode(action, sim.getState())
        #uctTree.setRootNode(sim.getState(), sim.getActions(), r, sim.isTerminal())

        logging.info('state ' + str(env.state))

        traj.append(env.state)

        step += 1

    traj = np.array(traj)

    return traj, {}
