import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear, MSELoss
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.utils.data.sampler import SubsetRandomSampler
from topo_data import Autopo, split_balance_data
import numpy as np
import math
import csv
from scipy import stats
from easydict import EasyDict
import argparse



class GraphInteractionLayer(nn.Module):

    def __init__(self,
                 n_node_info,
                 n_edge_attr,
                 n_edge_code, ):
        super(GraphInteractionLayer, self).__init__()

        self.edge_processor = nn.Linear(n_edge_attr + n_node_info * 2, n_edge_code)
        self.node_processor = nn.Linear(n_node_info + n_edge_code, n_node_info-4)

    def forward(self, node_info, edge_attr, adj, return_edge_code=False):
        """
        :param return_edge_code: whether return [edge_code]
        :param node_code: B x N x D1
        :param node_attr: B x N x D2
        :param edge_attr: B x N x N x D3
        :param adj: B x N x N
        :return: new_node_code: B x N x D1
        """

        B, N = node_info.size(0), node_info.size(1)

        receiver_info = node_info[:, :, None, :].repeat(1, 1, N, 1)
        # [256,7,7,104]
        sender_info = node_info[:, None, :, :].repeat(1, N, 1, 1)
        # [256,7,7,104]
        edge_input = torch.cat([edge_attr, receiver_info, sender_info], 3)
        # [256,7,7,211]
        edge_code = F.leaky_relu(self.edge_processor(edge_input.reshape(B * N * N, -1)).reshape(B, N, N, -1))
        # [256,7,7,100]

        edge_agg = (edge_code * adj[:, :, :, None]).sum(2)
        # [256,7,100]
        node_input = torch.cat([node_info, edge_agg], 2)
        # [256,7,204]
        new_node_code = self.node_processor(node_input.reshape(B * N, -1)).reshape(B, N, -1)
        # [256,7,100]

        if return_edge_code: return new_node_code, edge_code

        return new_node_code


class GIN(nn.Module):
    """
    Graph Interaction Network
    """

    def __init__(self,
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code,
                 n_layers=1,
                 use_gpu=False,
                 dropout=0):
        super(GIN, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer = GraphInteractionLayer(n_node_info=n_node_attr+n_node_code,
                                          n_edge_attr=n_edge_attr,
                                          n_edge_code=n_edge_code)
            self.layers.append(layer)
            setattr(self, 'gin_layer_{}'.format(i), layer)


        # 将node_attr进行encode
        self.n_node_code = n_node_code
        self.n_layers = n_layers
        self.use_gpu = use_gpu

        if dropout > 0:
            self.drop_layers = [nn.Dropout(p=dropout)] * n_layers
        else:
            self.drop_layers = None

    def forward(self, x, node_attr, edge_attr, adj, return_edge_code=False):

        # node_info = torch.cat([x, node_attr], 2)
        # print("n_layers: ",self.n_layers)
        x_edge_codes = []
        for i in range(self.n_layers):
            if return_edge_code:
                x, x_edge = self.layers[i](node_info, edge_attr, adj, return_edge_code)
                x_edge_codes.append(x_edge)
            else:
                node_info = torch.cat([x, node_attr], 2)
                x = self.layers[i](node_info, edge_attr, adj)
            x = F.leaky_relu(x)
            if self.drop_layers is not None:
                x = self.drop_layers[i](x)

        if return_edge_code:
            return x, x_edge_codes

        return x


# ======================================================================================================================
class PT_GNN(nn.Module):

    def __init__(self, args):
        super(PT_GNN, self).__init__()

        nhid = args.len_hidden
        self.gnn_encoder1 = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=1, use_gpu=False,
                               dropout=args.dropout)
        self.gnn_encoder2 = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                                n_edge_attr=args.len_edge_attr, n_layers=1, use_gpu=False,
                                dropout=args.dropout)

        self.lin1 = torch.nn.Linear(6 * nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, args.output_size)

        self.node_encoder = nn.Sequential(
            # TODO Where does args.len_node_attr send to
            nn.Linear(args.len_node_attr, nhid),
            nn.LeakyReLU(0.1),
        )

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj, gnn_layers = input

        x1 = self.node_encoder(node_attr)
        # print("gnn_layers: ", gnn_layers)
        for i in range(gnn_layers):
            x1 = self.gnn_encoder1(x1, node_attr, edge_attr1, adj)
        gnn_node_codes1 = x1
        x2 = self.node_encoder(node_attr)
        for i in range(gnn_layers):
            x2 = self.gnn_encoder2(x2, node_attr, edge_attr2, adj)
        gnn_node_codes2 = x2

        gnn_node_codes=torch.cat([gnn_node_codes1,gnn_node_codes2],dim=2)

        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, 1, :], gnn_node_codes[:, 2, :]], 1)
        # print("gnn_code", gnn_code.shape)

        x = self.lin1(gnn_code)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        pred = self.output(x)

        return torch.sigmoid(pred)


class Serial_GNN(nn.Module):
    def __init__(self, args):
        super(Serial_GNN, self).__init__()

        nhid = args.len_hidden

        self.gnn_encoder1 = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=1, use_gpu=False,
                               dropout=args.dropout)
        self.gnn_encoder2 = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=1, use_gpu=False,
                               dropout=args.dropout)

        self.lin1 = torch.nn.Linear(6 * nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, args.output_size)

        self.node_encoder = nn.Sequential(
            #TODO Where does args.len_node_attr send to
            nn.Linear(args.len_node_attr, nhid),
            nn.LeakyReLU(0.1),
        )

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj, gnn_layers = input
        x1 = self.node_encoder(node_attr)
        for i in range(gnn_layers):
            x1 = self.gnn_encoder1(x1, node_attr, edge_attr1, adj)
        x2 = x1
        for i in range(gnn_layers):
            x2 = self.gnn_encoder2(x2, node_attr, edge_attr2, adj)

        gnn_node_codes = torch.cat([x1,x2],dim=2)
        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, 1, :], gnn_node_codes[:, 2, :]], 1)

        x = self.lin1(gnn_code)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        pred = self.output(x)

        return torch.sigmoid(pred)



class LOOP_GNN(nn.Module):

    def __init__(self, args):
        super(LOOP_GNN, self).__init__()

        nhid = args.len_hidden

        self.gnn_encoder1 = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=1, use_gpu=False,
                               dropout=args.dropout)
        self.gnn_encoder2 = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=1, use_gpu=False,
                               dropout=args.dropout)

        self.lin1 = torch.nn.Linear(6 * nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, args.output_size)

        self.node_encoder = nn.Sequential(
            #TODO Where does args.len_node_attr send to
            nn.Linear(args.len_node_attr, nhid),
            nn.LeakyReLU(0.1),
        )
        #input:4,output:100



    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj, gnn_layers = input

        x = self.node_encoder(node_attr)

        for loops_num in range(gnn_layers):
            x1 = self.gnn_encoder1(x, node_attr, edge_attr1, adj)
            x = self.gnn_encoder2(x1, node_attr, edge_attr2, adj)

        gnn_node_codes = gnn_node_codes = torch.cat([x1,x],dim=2)
        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, 1, :], gnn_node_codes[:, 2, :]], 1)
        # print("gnn_code", gnn_code.shape)

        x = self.lin1(gnn_code)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        pred = self.output(x)

        return torch.sigmoid(pred)





