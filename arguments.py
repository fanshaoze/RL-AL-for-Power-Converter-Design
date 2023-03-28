import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--no_cuda', action='store_true', default=False, help='do not use cuda')

    parser.add_argument(
        '--query-times', type=int, default=1, help='the number of queries')
    parser.add_argument(
        '--sigma', type=float, default=1e-4, help='likelihood noise')

    parser.add_argument(
        '--num-runs', type=int, help='number of runs for UCT')

    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument(
        '--seed-range', nargs='+', type=int, default=[0, 2], help='random seed range')

    parser.add_argument(
        '--dry', action='store_true', default=False, help='dry run')

    parser.add_argument(
        '--debug', action='store_true', default=False, help='debug mode')

    parser.add_argument(
        '--skip-sim', action='store_true', default=False, help='skip the the topologies which is not in the '
                                                               'simulation hash')

    parser.add_argument(
        '--sweep', action='store_true', default=True, help='sweep duty cycles')
    parser.add_argument(
        '--get-traindata', action='store_true', default=False, help='want to get training data using uct')

    parser.add_argument(
        '--k-list', nargs='+', type=int, default=[3], help='evaluate top k topos'
    )
    parser.add_argument(
        '--output', type=str, default='result', help='output json file name'
    )

    parser.add_argument(
        '--model', type=str, default='analytics', choices=['simulator', 'transformer', 'gp', 'analytics', 'gnn'],
        help='surrogate model'
    )
    parser.add_argument(
        '--traj', nargs='+', type=int, default=[2, 3], help='trajectory numbers'
    )

    parser.add_argument(
        '--eff-model', type=str, default='reg_eff-3-5', help='eff pt model file name'
    )
    parser.add_argument(
        '--vout-model', type=str, default='reg_vout-4-5', help='vout pt model file name'
    )
    parser.add_argument(
        '--reward-model', type=str, default=None, help='reward pt model file name'
    )
    parser.add_argument(
        '--vocab', type=str, default='dataset_5_vocab.json', help='transformer vocab file'
    )
    parser.add_argument(
        '--round', type=str, default='None', help='using classified vout'
    )
    parser.add_argument('--gnn-nodes', type=int, default=40, help='number of nodes in hidden layer in GNN')
    parser.add_argument('--predictor-nodes', type=int, default=10, help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('--gnn-layers', type=int, default=4, help='number of layer')
    parser.add_argument('--model-index', type=int, default=3, help='model index')
    parser.add_argument('--nnode', type=int, default=7, help='number of node')


    args = parser.parse_args()

    return args
