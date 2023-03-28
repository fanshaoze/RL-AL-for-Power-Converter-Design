import argparse
import os


def single_top_extend(args):
    """
    if single value of the top, extend down to 1 or 1%
    @param args:
    @return:
    """
    if len(args.k_list) == 1:
        args.k_list = list(range(1, 1 + args.k_list[0]))
    if len(args.prop_top_ratios) == 1:
        args.prop_top_ratios = [m * 0.01 for m in range(1, 1 + int(args.prop_top_ratios[0] / 0.01))]
    return args


def get_args():
    """
    Get and update the arguments of topoQueryExp
    @return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed-range', nargs='+', type=int, default=[0, 2], help='random seed range')
    parser.add_argument(
        '--k-list', nargs='+', type=int, default=[1, 5, 10],
        help='evaluate top k topos, if len=1 then get top range(k_list[0])')
    parser.add_argument(
        '--prop-top-ratios', nargs='+', type=float, default=[0.05],
        help='prop top ratios, if len=1 then get top range(prop-top-ratios[0])')
    parser.add_argument(
        '--random-top', action='store_true', default=False, help='random get top')
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='gpu id')

    parser.add_argument(
        '--debug_traj', action='store_true', default=False, help='debug_traj')
    # parser.add_argument(
    #     '--traj-list', nargs='+', type=int, default=[[50]], help='trajectory numbers')
    base_trajs = [16, 8, 4, 2]
    parser.add_argument(
        '--traj-list', nargs='+', type=int, default=[[2 * traj for traj in base_trajs],
                                                     [7 * traj for traj in base_trajs],
                                                     [19 * traj for traj in base_trajs]], help='trajectory numbers')

    parser.add_argument(
        '--eff-model-seed', type=str, default='0', help='random seed for eff model for fix comp')
    parser.add_argument(
        '--vout-model-seed', type=str, default='0', help='random seed for vout model for fix comp')
    parser.add_argument(
        '--eff-model-seeds', nargs='+', type=int, default=[1], help='random seed for eff model for fix comp')
    parser.add_argument(
        '--vout-model-seeds', nargs='+', type=int, default=[1], help='random seed for vout model for fix comp')
    parser.add_argument(
        '--training-ratio-str', type=str, default="16_0.075", help='training data file name')
    parser.add_argument(
        '--training-data-file', type=str,
        default="transformer_SVGP/data/dataset_5_cleaned_label_train_0.075_16.json", help='training data file')
    parser.add_argument(
        '--dev-file', type=str, default='transformer_SVGP/data/dataset_5_cleaned_label_train_0.075_sample_dev.json',
        help='dev set used for retraining transformer')
    parser.add_argument(
        '--test-file', type=str, default='transformer_SVGP/data/dataset_5_cleaned_label_train_0.075_sample_test.json',
        help='test set used for retraining transformer')
    parser.add_argument(
        '--model', type=str, choices=['simulator', 'transformer', 'gp', 'analytics', 'gnn'], default='transformer',
        help='surrogate model')

    # active learning arguments
    parser.add_argument(
        '--debug-AL', action='store_true', default=True, help='query and update rewards')
    parser.add_argument(
        '--update-rewards', action='store_true', default=True, help='query and update rewards')
    parser.add_argument(
        '--AL-strategy', type=str, default='hybrid', choices=['mean', 'uncertainty', 'diversity', 'hybrid'],
        help='active learning query strategy')
    parser.add_argument(
        '--save-realR', action='store_true', default=True,
        help='save the ground truth rewards for the topo and following queries')
    parser.add_argument(
        '--cumulate-candidates', action='store_true', default=True, help='cumulate the candidate states for each step')
    parser.add_argument(
        '--update-gp', action='store_true', default=False, help='retrain surrogate model after query')
    parser.add_argument(
        '--reuse-tree-after-al', action='store_true', default=True, help='reuse uct trees after AL')
    parser.add_argument(
        '--recompute-tree-rewards', action='store_true', default=True, help='recompute the rewards in tree after AL')
    parser.add_argument(
        '--replan-times', type=int, default=1, help='planner replan times to generate each data point')
    # # for run
    parser.add_argument(
        '--epoch', type=int, default=3, help='max epoch step')  # candidates are 5, 10, 50, 100, or inf (like 10000)
    parser.add_argument(
        '--patience', type=int, default=2, help='max patience')
    parser.add_argument(
        '--sample-ratio', type=int, default=0.0001, help='random sample ratio from old dataset')
    parser.add_argument(
        '--retrain-query-ratio', type=int, default=0.1, help='sample ratio from candidate states for retraining')

    # UCT specific parameters
    parser.add_argument(
        '--component-default-policy', action='store_true', default=True, help='component default policy')
    parser.add_argument(
        '--path-default-policy', action='store_true', default=True, help='path default policy')
    parser.add_argument(
        '--algorithm', type=str, default='UCT', help='ALG to generate circuit')
    parser.add_argument(
        '--sweep', action='store_true', default=False, help='sweep duty cycles')
    parser.add_argument(
        '--skip-sim', action='store_true', default=False, help='skip the the circuits not in the simulation hash')

    # most unchanged
    parser.add_argument(
        '--no_cuda', action='store_true', default=False, help='do not use cuda')
    parser.add_argument(
        '--query-times', type=int, default=1, help='the number of queries')
    parser.add_argument(
        '--sigma', type=float, default=1e-4, help='likelihood noise')
    parser.add_argument(
        '--num-runs', type=int, help='number of runs for UCT')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 0)')  # dummy
    parser.add_argument(
        '--dry', action='store_true', default=False, help='dry run')
    parser.add_argument(
        '--debug', action='store_true', default=False, help='debug mode')

    parser.add_argument(
        '--get-traindata', action='store_true', default=False, help='want to get training data using uct')
    # parser.add_argument(
    #     '--top-ratio', nargs='+', type=float, default=0.09, help='ratio to get top k topos'
    # )
    parser.add_argument(
        '--output', type=str, default='result', help='output json file name')

    parser.add_argument(
        '--use_external_expression_hash', action='store_true', default=False, help='use external expression hash')
    parser.add_argument(
        '--using_exp_inner_hash', action='store_true', default=False, help='using_exp_inner_hash')
    parser.add_argument(
        '--save_expression', action='store_true', default=False, help='save_expression')
    parser.add_argument(
        '--save_simu_results', action='store_true', default=False, help='save_simu_results')

    # parser.add_argument(
    #     '--traj-list', nargs='+', type=int, default=[[210], [560], [630]], help='trajectory numbers'
    # )

    # parser.add_argument(
    #     '--traj-list', nargs='+', type=int, default=[[35, 35], [70, 70], [105, 105], [140, 140]], help='trajectory numbers'
    # )

    parser.add_argument(
        '--vocab', type=str, default='transformer_SVGP/compwise_vocab.json', help='transformer vocab file')
    parser.add_argument(
        '--model-prefix', type=str, default='transformer_SVGP/lstm_model/')
    parser.add_argument(
        '--reward-model', type=str, default='analytics', help='reward pt model file name')
    parser.add_argument(
        '--round', type=str, default='None', help='using classified vout')
    parser.add_argument('--gnn-nodes', type=int, default=40, help='number of nodes in hidden layer in GNN')
    parser.add_argument('--predictor-nodes', type=int, default=10,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('--gnn-layers', type=int, default=4, help='number of layer')
    parser.add_argument('--model-index', type=int, default=3, help='model index')
    parser.add_argument('--nnode', type=int, default=7, help='number of node')

    args = single_top_extend(parser.parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    if len(args.eff_model_seeds) == 1:
        args.eff_model_seeds = [i for i in range(args.eff_model_seeds[0])]
    if len(args.vout_model_seeds) == 1:
        args.vout_model_seeds = [i for i in range(args.vout_model_seeds[0])]

    postfix = '' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu_'
    args.eff_model = args.model_prefix + '5_all_lstm_gp_eff_' + postfix + \
                     args.training_ratio_str + '_' + str(args.eff_model_seed) + '.pt'
    args.vout_model = args.model_prefix + '5_all_lstm_gp_vout_' + postfix + \
                      args.training_ratio_str + '_' + str(args.vout_model_seed) + '.pt'

    args.eff_models = [args.model_prefix + '5_all_lstm_gp_eff_' + postfix + \
                       args.training_ratio_str + '_' + str(eff_model_seed) + '.pt'
                       for eff_model_seed in args.eff_model_seeds]
    args.vout_models = [args.model_prefix + '5_all_lstm_gp_vout_' + postfix + \
                        args.training_ratio_str + '_' + str(vout_model_seed) + '.pt'
                        for vout_model_seed in args.vout_model_seeds]
    # args.eff_models = [args.model_prefix + '5_all_lstm_eff_' + postfix + \
    #                    args.training_ratio_str + '_' + str(eff_model_seed) + '.pt'
    #                    for eff_model_seed in [1, 1]]
    # args.vout_models = [args.model_prefix + '5_all_lstm_vout_' + postfix + \
    #                     args.training_ratio_str + '_' + str(vout_model_seed) + '.pt'
    #                     for vout_model_seed in [1, 1]]

    return args
