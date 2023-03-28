import logging
import random
import time

import numpy as np

from algs.activeLearning import ucb, random_select_state
from topoUtils import find_paths, get_topo_key
from topo_analysis.topoGraph import TopoGraph
from PM_GNN.code.gen_topo_for_dateset import convert_to_netlist
from transformer_SVGP.transformer_utils import evaluate_eff_vout_model
from transformer_SVGP.test_model import test
from transformer_SVGP.GetReward import compute_batch_reward


def simple_reward_std(eff_std, vout_std):
    return eff_std + vout_std / 50.


# TODO how to compute this std??
def combined_reward_std(eff_std, vout_std):
    pass


reward_std = simple_reward_std


def find_opt_in_k_candidates(sim, cand_states, k=1):
    """
    Find the ground-truth optimal topology from the top k predicted by the surrogate model.
    :return: the index in cand_states
    """
    surrogate_rewards = np.array([sim.get_reward(state) for state in cand_states])
    # k topologies with highest surrogate rewards
    candidate_indices = surrogate_rewards.argsort()[-k:]

    index_to_true_reward = {idx: sim.get_true_reward(cand_states[idx]) for idx in candidate_indices}
    # the top one (reward, eff, vout) in the set above
    opt_state_index = max(candidate_indices, key=lambda idx: index_to_true_reward[idx])

    return opt_state_index


# def query_update(state_list, sim, factory, update_gp=False, query_times=1, random_query_times=0,
#                  strategy='ucb'):
#     """
#     Query and then only update rewards of states, no generalization using gp.
#     """
#     assert len(state_list), 'Candidate state list for querying cannot be empty.'
#
#     # get means and stds of all states in state_list
#     means = [sim.get_reward(state) for state in state_list]
#     # if a state is queried, then its std is 0, otherwise compute using reward_std function
#     stds = [reward_std(sim.get_surrogate_eff_std(state),
#                        sim.get_surrogate_vout_std(state))
#             for state in state_list]
#
#     queried_states = []
#     true_effs = []
#     true_vouts = []
#     # use usb to select candidate state
#     for query_time in range(query_times):
#         if strategy == 'ucb':
#             bandit_index = ucb(means=means, stds=stds, query_time=query_time + 1)
#         elif strategy == 'mean':
#             bandit_index = np.argmax(means)
#         elif strategy == 'var':
#             bandit_index = np.argmax(stds)
#         elif strategy == 'random':
#             bandit_index = random.choice(range(len(state_list)))
#         else:
#             raise Exception('unknown query strategy')
#
#         queried_state = state_list[bandit_index]
#         logging.info('query bandit ' + str(bandit_index))
#
#         response_reward, true_eff, true_vout = sim.get_true_performance(queried_state)
#         logging.info('response reward ' + str(response_reward))
#
#         queried_states.append(queried_state)
#         true_effs.append(true_eff)
#         true_vouts.append(true_vout)
#         print('number:', bandit_index, 'changes from ', means[bandit_index], ' to ', response_reward)
#         means[bandit_index] = response_reward
#         stds[bandit_index] = 0
#     # use random to add candidate state
#     for query_time in range(random_query_times):
#         logging.info('query time ' + str(query_time))
#
#         bandit_index = random_select_state(means=means, stds=stds, query_time=query_time + 1)
#
#         queried_state = state_list[bandit_index]
#         logging.info('query bandit ' + str(bandit_index))
#
#         response_reward, true_eff, true_vout = sim.get_true_performance(queried_state)
#         logging.info('response reward ' + str(response_reward))
#
#         queried_states.append(queried_state)
#         true_effs.append(true_eff)
#         true_vouts.append(true_vout)
#
#         means[bandit_index] = response_reward
#         stds[bandit_index] = 0
#     effi_early_stop, vout_early_stop = 0, 0
#     if update_gp:
#         # retrain the model
#         # effi_early_stop, vout_early_stop = factory.add_data_to_model_and_train(
#         #     path_set=[find_paths(state) for state in queried_states],
#         #     effs=true_effs,
#         #     vouts=true_vouts)
#         effi_early_stop, vout_early_stop = factory.add_data_to_model_and_train(
#             path_set=[find_paths(state) for state in queried_states],
#             duties=[state.parameters[0] for state in queried_states],
#             effs=true_effs,
#             vouts=true_vouts)
#
#         # update the sim with retrained models
#         factory.update_sim_models(sim)
#
#     return means, effi_early_stop, vout_early_stop, queried_states
def get_top_diversity_indices(sim, state_list):
    return []


def select_inds(sort_inds, select_count, selected_inds):
    """
    select select_count inds from the sort_inds without overlap with selected_ind
    :return:
    """
    query_indices = []
    for ind in sort_inds:
        if select_count > 0:
            if ind not in selected_inds:
                query_indices.append(ind)
                select_count -= 1
        else:
            break
    return query_indices


def select_using_uncertainty(query_cand_indices, eff_uncertainty_var, vout_uncertainty_var, un_count, un_eff_ratio):
    eff_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: eff_uncertainty_var[i], reverse=True),
        select_count=int(un_count * un_eff_ratio),
        selected_inds=[])
    vout_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: vout_uncertainty_var[i], reverse=True),
        select_count=un_count - int(un_count * un_eff_ratio),
        selected_inds=eff_query_indices)
    return eff_query_indices + vout_query_indices


def select_using_prediction(query_cand_indices, reward_ensemble_predictions, pred_count, low_pred_ratio, selected_inds):
    low_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: reward_ensemble_predictions[i], reverse=False),
        select_count=int(pred_count * low_pred_ratio),
        selected_inds=selected_inds)
    high_query_indices = select_inds(
        sort_inds=sorted(query_cand_indices, key=lambda i: reward_ensemble_predictions[i], reverse=True),
        select_count=pred_count - int(pred_count * low_pred_ratio),
        selected_inds=selected_inds + low_query_indices)
    return high_query_indices + low_query_indices


def hybrid_query_strategy(query_cand_indices, reward_ensemble_predictions,
                          eff_uncertainty_var, vout_uncertainty_var, retrain_query_count, un_ratio, un_eff_ratio,
                          low_pred_ratio):
    query_indices = []
    un_count, pred_count = int(retrain_query_count * un_ratio), retrain_query_count - int(
        retrain_query_count * un_ratio)
    query_indices += select_using_uncertainty(query_cand_indices, eff_uncertainty_var, vout_uncertainty_var,
                                              un_count, un_eff_ratio)
    query_indices += select_using_prediction(query_cand_indices, reward_ensemble_predictions,
                                             pred_count, low_pred_ratio, selected_inds=query_indices)
    return query_indices


def get_EPU_ensemble_rewards(eff_ensemble_predictions, eff_uncertainty_stds,
                             vout_ensemble_predictions, vout_uncertainty_stds,
                             eff_un_ratio, vout_un_ratio, target_vout):
    """
    compute EPU based reward value
    @param vout_un_ratio:
    @param eff_un_ratio:
    @param eff_ensemble_predictions:
    @param eff_uncertainty_stds:
    @param vout_ensemble_predictions:
    @param vout_uncertainty_stds:
    @param target_vout:
    @return:
    """
    assert len(eff_ensemble_predictions) == len(eff_uncertainty_stds) == \
           len(vout_ensemble_predictions) == len(vout_uncertainty_stds)
    inds = [i for i in range(len(eff_ensemble_predictions))]
    epu_ensemble_effs = [min(eff_ensemble_predictions[i] + eff_un_ratio * eff_uncertainty_stds[i], 1.) for i in inds]
    epu_ensemble_vouts = []
    for i in inds:
        if float(vout_ensemble_predictions[i]) < target_vout:
            epu_ensemble_vouts.append(min(vout_ensemble_predictions[i] + vout_un_ratio * vout_uncertainty_stds[i],
                                          target_vout))
        else:
            epu_ensemble_vouts.append(max(vout_ensemble_predictions[i] - vout_un_ratio * vout_uncertainty_stds[i],
                                          target_vout))
    return compute_batch_reward(epu_ensemble_effs, epu_ensemble_vouts, target_vout=target_vout)


def query_update(state_list, ensemble_infos, queried_state_keys, sim, factory, update_gp=False, query_times=1,
                 strategy='mean'):
    """
    Query and then only update rewards of states, no generalization using gp.
    """
    assert len(state_list), 'Candidate state list for querying cannot be empty.'

    # get means and stds of all states in state_list
    query_cand_indices = [i for i in range(len(state_list))
                          if
                          get_topo_key(state_list[i]) + '$' + str(state_list[i].parameters) not in queried_state_keys]

    # TODO batch
    start_time = time.time()
    reward_ensemble_predictions, eff_ensemble_predictions, eff_uncertainty_stds, \
    vout_ensemble_predictions, vout_uncertainty_stds = \
        ensemble_infos if ensemble_infos else sim.sequential_generate_ensemble_infos(state_list)

    print(f"get ensemble info time:{time.time() - start_time}")
    # if a state is queried, then its std is 0, otherwise compute using reward_std function
    # uncertainty = [reward_std(sim.get_surrogate_eff_std(state), sim.get_surrogate_vout_std(state))
    #                for state in state_list]
    # Diversity = get_top_diversity_indices(sim=sim, state_list=state_list)

    # upper = [means[i] + uncertainty[i] for i in range(len(state_list))]

    true_effs = []
    true_vouts = []
    valids = []
    rewards = []

    if strategy == 'mean':
        vout_un_ratio, eff_un_ratio = 0.0, 0.0
    elif strategy == 'uncertainty':
        # TODO: give a glance
        vout_un_ratio, eff_un_ratio = 100, 100
    elif strategy == 'hybrid':
        vout_un_ratio, eff_un_ratio = 1, 1
    else:
        raise Exception('unknown query strategy')

    EPU_ensemble_rewards = get_EPU_ensemble_rewards(eff_ensemble_predictions, eff_uncertainty_stds,
                                                    vout_ensemble_predictions, vout_uncertainty_stds,
                                                    eff_un_ratio=eff_un_ratio, vout_un_ratio=vout_un_ratio,
                                                    target_vout=sim.configs_['target_vout'])

    # indices of states that we consider querying (eliminate ones that have been queried before)

    # select the indices of the top query_times
    # query_cand_indices: index, lambda: get scores[i] for i
    query_indices = sorted(query_cand_indices, key=lambda i: EPU_ensemble_rewards[i], reverse=True)[:query_times]
    # query_indices = hybrid_query_strategy(query_cand_indices, reward_ensemble_predictions,
    #                                       eff_uncertainty_stds, vout_uncertainty_stds,
    #                                       retrain_query_count=query_times,
    #                                       un_ratio=un_ratio, un_eff_ratio=un_eff_ratio, low_pred_ratio=low_pred_ratio)

    logging.info(f"queried indices are {query_indices}")

    query_states = [state_list[i] for i in query_indices]

    for i, state in zip(query_indices, query_states):
        response_reward, true_eff, true_vout = sim.get_true_performance(state)
        true_effs.append(true_eff)
        true_vouts.append(true_vout)
        rewards.append(response_reward)
        # if true_vouts == 0.0 and true_eff == 0:
        # TODO: we can add the valid condition here
        valids.append(1)
        print('Changing number:', i, ' from ', reward_ensemble_predictions[i], ' to ', response_reward)
        reward_ensemble_predictions[i] = response_reward

    effi_early_stop, vout_early_stop, epoch_i_eff, epoch_i_vout = 0, 0, 0, 0
    if update_gp:
        # retrain the model
        path_set = [find_paths(q_state) for q_state in query_states]
        duties = [q_state.parameters[0] for q_state in query_states]

        effi_early_stop, vout_early_stop, epoch_i_eff, epoch_i_vout = factory.add_data_to_model_and_train(
            path_set=path_set,
            duties=duties,
            effs=true_effs,
            vouts=true_vouts,
            valids=valids,
            rewards=rewards)

        # update the sim with retrained models
        factory.update_sim_models(sim)

    queried_state_keys.update(get_topo_key(state) + '$' + str(state.parameters) for state in query_states)

    return reward_ensemble_predictions, effi_early_stop, vout_early_stop, queried_state_keys, epoch_i_eff, epoch_i_vout
