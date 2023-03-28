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

# ======================================================================================================================
'''
GNN
'''


class GraphInteractionLayer(nn.Module):

    def __init__(self,
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code, ):
        super(GraphInteractionLayer, self).__init__()

        self.edge_processor = nn.Linear(n_edge_attr + (n_node_attr + n_node_code) * 2, n_edge_code)
        self.node_processor = nn.Linear(n_node_attr + n_node_code + n_edge_code, n_node_code)

    def forward(self, node_code, node_attr, edge_attr, adj, return_edge_code=False):
        """
        :param return_edge_code: whether return [edge_code]
        :param node_code: B x N x D1
        :param node_attr: B x N x D2
        :param edge_attr: B x N x N x D3
        :param adj: B x N x N
        :return: new_node_code: B x N x D1
        """

        B, N = node_code.size(0), node_code.size(1)

        node_info = torch.cat([node_code, node_attr], 2)

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
                 n_layers=2,
                 use_gpu=False,
                 dropout=0):
        super(GIN, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer = GraphInteractionLayer(n_node_attr=n_node_attr, n_node_code=n_node_code, n_edge_attr=n_edge_attr,
                                          n_edge_code=n_edge_code)
            self.layers.append(layer)
            setattr(self, 'gin_layer_{}'.format(i), layer)

        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_attr, n_node_code),
            nn.LeakyReLU(0.1),
        )
        # 将node_attr进行encode
        self.n_node_code = n_node_code
        self.n_layers = n_layers
        self.use_gpu = use_gpu

        if dropout > 0:
            self.drop_layers = [nn.Dropout(p=dropout)] * n_layers
        else:
            self.drop_layers = None

    def forward(self, node_attr, edge_attr, adj, return_edge_code=False):

        x = self.node_encoder(node_attr)
        x_edge_codes = []
        for i in range(self.n_layers):
            if return_edge_code:
                x, x_edge = self.layers[i](x, node_attr, edge_attr, adj, return_edge_code)
                x_edge_codes.append(x_edge)
            else:
                x = self.layers[i](x, node_attr, edge_attr, adj)
            x = F.leaky_relu(x)
            if self.drop_layers is not None:
                x = self.drop_layers[i](x)

        if return_edge_code:
            return x, x_edge_codes

        return x


# ======================================================================================================================


class Serial_GNN_Decoder(nn.Module):

    def __init__(self, args):
        super(Serial_GNN_Decoder, self).__init__()

        nhid = args.len_hidden
        self.gnn_encoder = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=args.gnn_layers, use_gpu=False,
                               dropout=args.dropout)
        # -------------------------------------------------------------#
        self.node_decoder = nn.Sequential(
            nn.Linear(nhid, args.len_node_attr),
            nn.LeakyReLU(0.1),
        )
        # -------------------------------------------------------------#
        self.lin1 = torch.nn.Linear(7 * nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj = input
        # print("PT_GNN input", len(input))
        # print("PT_GNN node_attr", node_attr.shape)
        # print("PT_GNN edge_attr1", edge_attr1.shape)
        # print("PT_GNN edge_attr2", edge_attr2.shape)
        # print("PT_GNN adi", adj.shape)

        # -----------------------------------------------------------------------------------#
        gnn_node_codes1 = self.gnn_encoder(node_attr, edge_attr1, adj)
        node_attr_phase1 = self.node_decoder(gnn_node_codes1)
        gnn_node_codes = self.gnn_encoder(node_attr_phase1, edge_attr2, adj)
        # -----------------------------------------------------------------------------------#
        #        gnn_node_codes1 = self.gnn_encoder(node_attr, edge_attr1, adj)
        #        gnn_node_codes2 = self.gnn_encoder(node_attr, edge_attr2, adj)
        #        gnn_node_codes=torch.cat([gnn_node_codes1,gnn_node_codes2],dim=2)
        # -----------------------------------------------------------------------------------#

        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, 1, :], gnn_node_codes[:, 2, :], gnn_node_codes[:, 3, :],
             gnn_node_codes[:, 4, :], gnn_node_codes[:, 5, :], gnn_node_codes[:, 6, :]], 1)
        # print("gnn_code", gnn_code.shape)

        x = self.lin1(gnn_code)
        x = self.lin2(x)
        pred = self.output(x)

        return torch.sigmoid(pred)


