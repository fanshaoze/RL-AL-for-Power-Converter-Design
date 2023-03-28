import csv
import datetime
import json
import logging
import sys, os
import numpy as np
import torch
import time

import config
from arguments import get_args

sys.path.append(os.path.join(sys.path[0], 'topo_data_util'))
sys.path.append(os.path.join(sys.path[0], 'transformer_SVGP'))
sys.path.append(os.path.join(sys.path[0], 'PM_GNN/code'))

if config.task == 'uct_3_comp':
    sys.path.append(os.path.join(sys.path[0], 'UCFTopo_dev'))
    from UCFTopo_dev.main import main as run_uct
elif config.task == 'uct_5_comp':
    sys.path.append(os.path.join(sys.path[0], 'UCT_5_UCB_unblc_restruct_DP_v1'))
    from UCT_5_UCB_unblc_restruct_DP_v1.main import main as run_uct_5_comp
else:
    raise Exception('unknown task ' + config.task)

from al_util import feed_random_seeds

dir = os.path.dirname(__file__)


def gp_reward_uct_exp(args):
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_csv_header = ['effi', 'vout', 'reward', 'time', ' queries', 'avg_time', 'new_query']

    if args.model == 'simulator' or args.model == 'analytics':
        sim_init = None  # None forces uct to use NGSPice Simulator
    elif args.model == 'gp':
        from topo_envs.GPRewardSim import GPRewardTopologySim

        def sim_init(*a):
            return GPRewardTopologySim('efficiency.pt', 'vout.pt', args.debug, *a)
    elif args.model == 'transformer':
        from topo_envs.TransformerRewardSim import TransformerRewardSimFactory

        factory = TransformerRewardSimFactory(
            eff_model_file=os.path.join(dir, args.eff_model),
            vout_model_file=os.path.join(dir, args.vout_model),
            vocab_file=os.path.join(dir, args.vocab),
            device=device, training_data=None)

        sim_init = factory.get_sim_init()
    elif args.model == 'gnn':
        from topo_envs.GNNRewardSim import GNNRewardSim

        def sim_init(*a):
            return GNNRewardSim(args.eff_model, args.vout_model, args.reward_model, args.debug, *a)

    else:
        raise Exception('unknown model ' + args.model)

    if args.seed_range is not None:
        seed_range = range(args.seed_range[0], args.seed_range[1])
    elif args.seed is not None:
        seed_range = [args.seed]
    else:
        # just run random seed 0 by default
        seed_range = [0]

    results = {}
    for k in args.k_list:
        results[k] = {}
        for _traj_num in args.traj:
            results[k][_traj_num] = []
    rows = []  # record rewards after each query, size: num of query * num of random seed
    last_rows = []
    analysis_results = {}
    save_output_tops = {}
    for seed in seed_range:
        logging.info('seed ' + str(seed))
        for traj_num in args.traj:
            logging.info('traj num ' + str(traj_num))
            feed_random_seeds(seed)

            start_time = time.time()
            if config.task == 'uct_3_comp':
                info = run_uct(Sim=sim_init, traj=traj_num, args_file_name='UCFTopo_dev/config')
            elif config.task == 'uct_5_comp':
                from UCT_5_UCB_unblc_restruct_DP_v1.config import uct_configs
                if args.model == 'analytics' or args.model == 'simulator':
                    # the seed range in GNN and analytics or simulator is different
                    # in GNN, we use the real seeds to call run_CUT_5_comp
                    # in analytics or simulator, we just use the length of seed_range as the test number,
                    # just call run_uct_5_comp just once, the test number will let the UCT run multiple times,
                    uct_configs['test_number'] = len(seed_range)
                    uct_configs['trajectories'] = args.traj
                    uct_configs['reward_method'] = args.model
                    uct_configs['round'] = args.round
                    uct_configs['skip_sim'] = args.skip_sim
                    uct_configs['sweep'] = args.sweep
                    uct_configs['topk_list'] = args.k_list
                    uct_configs['get_traindata'] = args.get_traindata

                    info = run_uct_5_comp(Sim=sim_init, traj=None, configs=uct_configs)
                    return None

                else:
                    uct_configs['test_number'] = 1
                    uct_configs['reward_method'] = 'gnn'
                    uct_configs['skip_sim'] = args.skip_sim
                    uct_configs['sweep'] = args.sweep
                    uct_configs['topk_list'] = args.k_list
                    uct_configs['get_traindata'] = False
                    uct_configs['round'] = args.round
                    uct_configs['gnn_nodes'] = args.gnn_nodes
                    uct_configs['predictor_nodes'] = args.predictor_nodes
                    uct_configs['gnn_layers'] = args.gnn_layers
                    uct_configs['model_index'] = args.model_index
                    uct_configs['reward_model'] = args.reward_model
                    uct_configs['nnode'] = args.nnode

                    info = run_uct_5_comp(Sim=sim_init, traj=traj_num, configs=uct_configs)
            else:
                raise Exception('unknown task ' + config.task)

            sim = info['sim']
            # if args.sweep:
            cand_states = [topo[0] for topo in sim.no_isom_seen_state_list]
            logging.info('cand state size ' + str(len(cand_states)))
            query_num = info['query_num']

            if args.model == 'simulator':
                # sort all (reward, eff, vout) by rewards
                sorted_performances = sorted(sim.graph_2_reward.values(), key=lambda _: _[0])
            else:
                # for surrogate models
                surrogate_rewards = np.array([sim.get_reward(state) for state in cand_states])

            for k in args.k_list:
                logging.info('k = ' + str(k))
                if args.model == 'simulator':
                    top_k = sorted_performances[-k:]
                    top_1 = sorted_performances[-1]
                    # this is dummy for simulator
                    surrogate_top_k = []
                else:
                    # k topologies with highest surrogate rewards
                    candidate_indices = surrogate_rewards.argsort()[-k:]

                    surrogate_top_k = [(sim.get_reward(cand_states[idx]), sim.get_surrogate_eff(cand_states[idx]),
                                        sim.get_surrogate_vout(cand_states[idx]))
                                       for idx in candidate_indices]
                    logging.info('top k surrogate info: ' + str(surrogate_top_k))

                    # the true (reward, eff, vout) of top k topologies decided by the surrogate model
                    top_k = [sim.get_true_performance(cand_states[idx]) for idx in candidate_indices]
                    logging.info('top k true perf: ' + str(top_k))

                    topk_rewards = []
                    for top in top_k:
                        topk_rewards.append(top[0])
                    if traj_num in save_output_tops:
                        save_output_tops[traj_num].append(topk_rewards)
                    else:
                        save_output_tops[traj_num] = [topk_rewards]
                    # the top one (reward, eff, vout) in the set above
                    top_1 = max(top_k, key=lambda _: _[0])

                with open('./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/database/key-sim-result.json',
                          'w') as f:
                    json.dump(sim.key_sim_effi_, f)
                f.close()

                end_time = time.time()
                execution_time = round(end_time - start_time, 2)

                # TODO for debug 'surrogate_top_k': surrogate_top_k
                if sim.new_query_counter != 0:
                    results[k][traj_num].append([top_1[1], top_1[2], top_1[0], execution_time, query_num,
                                                 sim.new_query_time/sim.new_query_counter, sim.new_query_counter])
                else:
                    results[k][traj_num].append([top_1[1], top_1[2], top_1[0], execution_time, query_num, 0, 0])

            # save to file after each random seed
            with open(args.output + '.json', 'w') as f:
                json.dump(results, f)

    file_signal = str(os.getpid())

    merged_traj_results = {}
    seed_length = len(results[k][traj_num])
    for k in args.k_list:
        merged_traj_results[k] = []
        for i in range(seed_length):
            merged_traj_results[k].append([])
            headers = []
            for traj_num in args.traj:
                headers.extend(output_csv_header)
                merged_traj_results[k][i].extend(results[k][traj_num][i])
    for k in args.k_list:
        with open('Results/GNN-merged-3-comp_result_top_' + str(k) + '_' + file_signal + '.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerows(merged_traj_results[k])

    simu_top_file_name = "Results/GNN-save-topks-" + date_str + "-" + str(
        os.getpid()) + ".json"
    with open(simu_top_file_name, 'w') as f:
        json.dump(save_output_tops, f)

    return None


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename=args.output + '.log',
                        filemode='w',
                        level=logging.INFO)

    gp_reward_uct_exp(args)
