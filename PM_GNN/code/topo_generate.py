import os

import torch
from PM_GNN.code.ml_utils import initialize_model, optimize_reward
import argparse

import numpy as np

if __name__ == '__main__':

    # ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=10, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=100, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=100,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=3, help='number of layer')
    parser.add_argument('-model_index', type=int, default=1, help='model index')
    parser.add_argument('-threshold', type=float, default=0, help='classification threshold')

    args = parser.parse_args()

    batch_size = args.batch_size
    n_epoch = args.n_epoch
    th = args.threshold

    # ======================== Data & Model ==========================#

    test_loader = None

    nf_size = 4
    ef_size = 3
    nnode = 4
    if args.model_index == 0:
        ef_size = 6

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert os.path.exists('reg_eff.pt')
    assert os.path.exists('reg_vout.pt')

    eff_model_state_dict, data_loader = torch.load('reg_eff.pt')
    eff_model = initialize_model(args.model_index, args.gnn_nodes, args.predictor_nodes, args.gnn_layers, nf_size, ef_size,
                             device)
    eff_model.load_state_dict(eff_model_state_dict)

    vout_model_state_dict, data_loader = torch.load('reg_vout.pt')
    vout_model = initialize_model(args.model_index, args.gnn_nodes, args.predictor_nodes, args.gnn_layers, nf_size, ef_size,
                             device)
    vout_model.load_state_dict(vout_model_state_dict)

    """
    r_model_state_dict, data_loader = torch.load('reg_reward.pt')
    r_model = initialize_model(args.model_index, args.gnn_nodes, args.predictor_nodes, args.gnn_layers, nf_size, ef_size,
                               device)
    r_model.load_state_dict(r_model_state_dict)
    """

    results = []

    for seed in range(20):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        result = optimize_reward(data_loader, eff_model, vout_model, n_epoch, batch_size, nnode, args.model_index, False,device,th)
        results.append(result)

    results = np.array(results)
    mean_results = results.mean(axis=0)

    print(results)
    print(mean_results)

    np.savetxt('topo_gen_results.csv', mean_results, delimiter=',', fmt='%.6f')
