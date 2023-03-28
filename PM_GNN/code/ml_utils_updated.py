import torch
import torch.nn.functional as F

import numpy as np
import math

from easydict import EasyDict

#from model_updated import SEIRAL_GNN,LOOP_GNN,SEIRAL_GNN_4Input
from model_updated import SEIRAL_GNN, LOOP_GNN, SEIRAL_GNN_4Input
import copy

from reward_fn import compute_batch_reward


def rse(y, yt):
    assert (y.shape == yt.shape)

    var = 0
    m_yt = yt.mean()
    #    print(yt,m_yt)
    for i in range(len(yt)):
        var += (yt[i] - m_yt) ** 2

    mse = 0
    for i in range(len(yt)):
        mse += (y[i] - yt[i]) ** 2

    rse = mse / (var + 0.0000001)

    rmse = math.sqrt(mse / len(yt))

    #    print(rmse)

    return rse


def initialize_model(model_index, gnn_nodes, gnn_layers, pred_nodes, nf_size, ef_size, device):
    # 初始化模型
    args = EasyDict()
    args.len_hidden = gnn_nodes
    args.len_hidden_predictor = pred_nodes
    args.len_node_attr = nf_size
    args.len_edge_attr = ef_size
    args.gnn_layers = gnn_layers
    args.use_gpu = False
    args.dropout = 0

    if model_index == 1:
        model = SEIRAL_GNN(args).to(device)
        return model
    elif model_index == 2:
        model = LOOP_GNN(args).to(device)
        return model
    elif model_index == 3:
        model = SEIRAL_GNN_4Input(args).to(device)
        return model
    elif model_index == 4:
        model = Serial_GNN_Decoder(args).to(device)
        return model
    else:
        assert ("Invalid model")
    # 选择model


def train(train_loader, val_loader, model, n_epoch, batch_size, num_node, device, model_index, optimizer):
    train_perform = []

    min_val_loss = 100

    for epoch in range(n_epoch):

        ########### Training #################

        train_loss = 0
        n_batch_train = 0

        model.train()

        for i, data in enumerate(train_loader):
            data.to(device)
            L = data.node_attr.shape[0]
            B = int(L / num_node)
            #                 print(L,B,data.node_attr)
            node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
            if model_index == 0:
                edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
            else:
                edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
                edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

            adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])
            y = data.label

            n_batch_train = n_batch_train + 1
            optimizer.zero_grad()
            if model_index == 0:
                out = model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
            else:
                out = model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device)))

            out = out.reshape(y.shape)
            assert (out.shape == y.shape)
            loss = F.mse_loss(out, y.float())

            #                 loss=F.binary_cross_entropy(out, y.float())
            loss.backward()
            optimizer.step()

            train_loss += out.shape[0] * loss.item()

        if epoch % 1 == 0:
            print('%d epoch training loss: %.3f' % (epoch, train_loss / n_batch_train / batch_size))

            n_batch_val = 0
            val_loss = 0

            #                epoch_min=0
            model.eval()

            for data in val_loader:

                n_batch_val += 1

                data.to(device)
                L = data.node_attr.shape[0]
                B = int(L / num_node)
                node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
                if model_index == 0:
                    edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
                else:
                    edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
                    edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

                adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])
                y = data.label

                n_batch_train = n_batch_train + 1
                optimizer.zero_grad()
                if model_index == 0:
                    out = model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                else:
                    out = model(
                        input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device)))

                out = out.reshape(y.shape)
                assert (out.shape == y.shape)
                #                     loss=F.binary_cross_entropy(out, y.float())
                loss = F.mse_loss(out, y.float())
                val_loss += out.shape[0] * loss.item()
            val_loss_ave = val_loss / n_batch_val / batch_size

            if val_loss_ave < min_val_loss:
                model_copy = copy.deepcopy(model)
                print('lowest val loss', val_loss_ave)
                epoch_min = epoch
                min_val_loss = val_loss_ave

            if epoch - epoch_min > 5:
                print("training loss:", train_perform)
                return model_copy

        train_perform.append(train_loss / n_batch_train / batch_size)
    return model


def test(test_loader, model, num_node, model_index, device):
    model.eval()
    accuracy = 0
    n_batch_test = 0
    gold_list = []
    out_list = []
    analytic_list = []

    TPR = 0
    FPR = 0
    count = 0
    for data in test_loader:
        data.to(device)
        L = data.node_attr.shape[0]
        B = int(L / num_node)
        node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
        if model_index == 0:
            edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
        else:
            edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
            edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

        adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])
        y = data.label.cpu().detach().numpy()

        n_batch_test = n_batch_test + 1
        if model_index == 0:
            out = model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
        else:
            out = model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
        out = out.reshape(y.shape)
        assert (out.shape == y.shape)
        out = np.array([x for x in out])
        gold = np.array(y.reshape(-1))
        gold = np.array([x for x in gold])

        gold_list.extend(gold)
        out_list.extend(out)

        L = len(gold)
        rse_result = rse(out, gold)
        np.set_printoptions(precision=2, suppress=True)
        #
        out_round = [int(i > 0.8) for i in out]
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for i in range(len(gold)):
            if gold[i] == 1 and out_round[i] == 1:
                TP += 1
            elif gold[i] == 1 and out_round[i] == 0:
                FN += 1
            elif gold[i] == 0 and out_round[i] == 1:
                FP += 1
            else:
                TN += 1
    #             print(TP/(TP+FN),FP/(TN+FP))
    #             TPR+=TP/(TP+FN)
    #             FPR+=FP/(TN+FP)
    #             count+=1
    #        print("Average erro:",TPR/count,FPR/count)
    #        print("Predicted:",out[0:128])
    #        print("True     :",gold[0:128])
    #        print("Error    :",len([i for i in abs(out-gold) if i > 0.5])/len(out))
    print("Final RSE:", rse(np.reshape(out_list, -1), np.reshape(gold_list, -1)))


def evaluate_top_K(out, ground_truth, k):
    out = np.array(out)
    ground_truth = np.array(ground_truth)

    candidates = out.argsort()[-k:]

    # the ground truth values of the candidates
    candidate_gt = ground_truth[candidates]
    # the candidate that has the best true value
    candidate_arg_max = candidates[np.argmax(candidate_gt)]

    return max(ground_truth[candidates]), candidate_arg_max


def optimize_reward(test_loader, eff_model, vout_model,
                    n_epoch, batch_size, num_node, model_index, flag, device, th):
    n_batch_test = 0

    sim_rewards = []
    analytic_rewards = []
    gnn_rewards = []

    all_sim_eff = []
    all_sim_vout = []
    all_gnn_eff = []
    all_gnn_vout = []
    all_analytic_eff = []
    all_analytic_vout = []

    k_list = [1, 10, 20, 30, 50, 100]

    sim_opts = []
    analytic_performs = {k: [] for k in k_list}
    gnn_performs = {k: [] for k in k_list}

    for data in test_loader:
        data.to(device)
        L = data.node_attr.shape[0]
        B = int(L / num_node)
        node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
        if model_index == 0:
            edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
        else:
            edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
            edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

        adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])

        sim_eff = data.sim_eff.cpu().detach().numpy()
        sim_vout = data.sim_vout.cpu().detach().numpy()

        analytic_eff = data.analytic_eff.cpu().detach().numpy()
        analytic_vout = data.analytic_vout.cpu().detach().numpy()

        n_batch_test = n_batch_test + 1
        if model_index == 0:
            eff = eff_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
        else:
            eff = eff_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
            # r = r_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()

        gnn_eff = eff.squeeze(1)
        gnn_vout = vout.squeeze(1)
        # r = r.squeeze(1)

        all_sim_eff.extend(sim_eff)
        all_sim_vout.extend(sim_vout)

        all_gnn_eff.extend(gnn_eff)
        all_gnn_vout.extend(gnn_vout)

        all_analytic_eff.extend(gnn_eff)
        all_analytic_vout.extend(gnn_vout)

        sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
        analytic_rewards.extend(compute_batch_reward(analytic_eff, analytic_vout))
        gnn_rewards.extend(compute_batch_reward(gnn_eff, gnn_vout))
        # out_list.extend(r)

        # sim opt
        sim_opt = np.max(sim_rewards)
        sim_opt_idx = np.argmax(sim_rewards)

        print('sim: reward {}, eff {}, vout {}'
              .format(sim_opt, all_sim_eff[sim_opt_idx], all_sim_vout[sim_opt_idx]))
        sim_opts.append(sim_opt)

        for k in k_list:
            # analytic
            analytic, analytic_idx = evaluate_top_K(analytic_rewards, sim_rewards, k)
            print(
                'analytic {}: true reward {}, true eff {}, true vout {}, predicted reward {}, predicted eff {}, predicted vout {}'
                .format(k, analytic, all_sim_eff[analytic_idx], all_sim_vout[analytic_idx],
                        analytic_rewards[analytic_idx], all_analytic_eff[analytic_idx],
                        all_analytic_vout[analytic_idx]))
            analytic_performs[k].append(analytic)

            # gnn
            gnn, gnn_idx = evaluate_top_K(gnn_rewards, sim_rewards, k)
            print(
                'gnn {}: true reward {}, true eff {}, true vout {}, predicted reward {}, predicted eff {}, predicted vout {}'
                .format(k, gnn, all_sim_eff[gnn_idx], all_sim_vout[gnn_idx],
                        gnn_rewards[gnn_idx], all_gnn_eff[gnn_idx], all_gnn_vout[gnn_idx]))
            gnn_performs[k].append(gnn)

        print()

    np.set_printoptions(precision=2, suppress=True)

    print("GNN RSE:", rse(np.array(gnn_rewards), np.array(sim_rewards)))
    print("Analytic RSE:", rse(np.array(analytic_rewards), np.array(sim_rewards)))

    print('sim', sim_opts)
    print('analytic', analytic_performs)
    print('gnn', gnn_performs)

    return [sim_opts] + list(analytic_performs.values()) + list(gnn_performs.values())
