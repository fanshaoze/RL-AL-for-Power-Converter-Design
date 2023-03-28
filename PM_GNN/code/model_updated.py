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
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code, ):
        super(GraphInteractionLayer, self).__init__()

        #self.edge_processor = nn.Linear(n_edge_attr + (n_node_attr + n_node_code) * 2, n_edge_code)
        #self.edge_processor = nn.Linear(n_edge_attr + (n_node_code + n_node_code) * 2, n_edge_code)
        self.edge_processor = nn.Linear(n_edge_attr +  n_node_code * 2, n_edge_code)


        self.node_processor = nn.Linear(n_node_code + n_node_code + n_edge_code, n_node_code)

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
        # [256,7,7,200]
        sender_info = node_info[:, None, :, :].repeat(1, N, 1, 1)
        # [256,7,7,200]
        edge_input = torch.cat([edge_attr, receiver_info, sender_info], 3)
        # [256,7,7,403]
        edge_code = F.leaky_relu(self.edge_processor(edge_input.reshape(B * N * N, -1)).reshape(B, N, N, -1))
        # [256,7,7,100]
        edge_agg = (edge_code * adj[:, :, :, None]).sum(2)
        # [256,7,100]
        node_input = torch.cat([node_info, edge_agg], 2)
        # [256,7,300]
        new_node_code = self.node_processor(node_input.reshape(B * N, -1)).reshape(B, N, -1)
        # [256,7,100]

        # B, N = node_code.size(0), node_code.size(1)
        # node_info = node_code
        # receiver_info = node_info[:, :, None, :].repeat(1, 1, N, 1)
        # sender_info = node_info[:, None, :, :].repeat(1, N, 1, 1)
        # edge_input = torch.cat([edge_attr, receiver_info, sender_info], 3)
        # edge_code = F.leaky_relu(self.edge_processor(edge_input.reshape(B * N * N, -1)).reshape(B, N, N, -1))
        # edge_agg = (edge_code * adj[:, :, :, None]).sum(2)
        # node_input = torch.cat([node_info, edge_agg], 2)
        # new_node_code = self.node_processor(node_input.reshape(B * N, -1)).reshape(B, N, -1)


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
                 n_layers=6,
                 use_gpu=False,
                 dropout=0):
        super(GIN, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer = GraphInteractionLayer(n_node_attr=n_node_attr, n_node_code=n_node_code, n_edge_attr=n_edge_attr,
                                          n_edge_code=n_edge_code)
            self.layers.append(layer)
            setattr(self, 'gin_layer_{}'.format(i), layer)

        self.n_node_code = n_node_code
        self.n_layers = n_layers
        self.use_gpu = use_gpu

        if dropout > 0:
            self.drop_layers = [nn.Dropout(p=dropout)] * n_layers
        else:
            self.drop_layers = None

    def forward(self, node_attr, edge_attr, adj, return_edge_code=False):

        # x = self.node_encoder(node_attr)
        x = node_attr

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



class SEIRAL_GNN(nn.Module):

    def __init__(self, args):
        super(SEIRAL_GNN, self).__init__()

        nhid = args.len_hidden
        n_node_attr = args.len_node_attr
        n_node_code = nhid
        self.gnn_encoder = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=args.gnn_layers, use_gpu=False,
                               dropout=args.dropout)
        # -------------------------------------------------------------#
        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_attr, n_node_code),
            nn.LeakyReLU(0.1),
        )
        # -------------------------------------------------------------#
        self.lin1 = torch.nn.Linear(7 * nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj = input

        x = self.node_encoder(node_attr)
        x = self.gnn_encoder(x, edge_attr1, adj)
        x = self.gnn_encoder(x, edge_attr2, adj)
        gnn_node_codes = x


        gnn_code = torch.cat([gnn_node_codes[:, 0, :],gnn_node_codes[:, 1, :],gnn_node_codes[:, 2, :],gnn_node_codes[:, 3, :],
                              gnn_node_codes[:, 4, :],gnn_node_codes[:, 5, :],gnn_node_codes[:, 6, :]],1)
        # print("gnn_code", gnn_code.shape)

        x = self.lin1(gnn_code)
        x = self.lin2(x)
        pred = self.output(x)

        return torch.sigmoid(pred)

class LOOP_GNN(nn.Module):

    def __init__(self, args):
        super(LOOP_GNN, self).__init__()

        nhid = args.len_hidden
        n_node_attr = args.len_node_attr
        n_node_code = nhid
        self.gnn_encoder = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=args.gnn_layers, use_gpu=False,
                               dropout=args.dropout)
        # -------------------------------------------------------------#
        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_attr, n_node_code),
            nn.LeakyReLU(0.1),
        )
        # -------------------------------------------------------------#
        self.lin1 = torch.nn.Linear(2 * nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj = input

        x = self.node_encoder(node_attr)

        for i in range(6):
            x = self.gnn_encoder(x, edge_attr1, adj)
            print("x: ",x.shape)
            x = self.gnn_encoder(x, edge_attr2, adj)

        gnn_node_codes = x
        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, 1, :], gnn_node_codes[:, 2, :]], 1)


        x = self.lin1(gnn_code)
        x = self.lin2(x)
        pred = self.output(x)

        return torch.sigmoid(pred)


class GraphInteractionLayer_4input(nn.Module):

    def __init__(self,
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code, ):
        super(GraphInteractionLayer_4input, self).__init__()

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


class GIN_4input(nn.Module):
    """
    Graph Interaction Network
    """

    def __init__(self,
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code,
                 n_layers=6,
                 use_gpu=False,
                 dropout=0):
        super(GIN_4input, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer = GraphInteractionLayer_4input(n_node_attr=n_node_attr, n_node_code=n_node_code, n_edge_attr=n_edge_attr,
                                          n_edge_code=n_edge_code)
            self.layers.append(layer)
            setattr(self, 'gin_layer_{}'.format(i), layer)

        self.n_node_code = n_node_code
        self.n_layers = n_layers
        self.use_gpu = use_gpu

        if dropout > 0:
            self.drop_layers = [nn.Dropout(p=dropout)] * n_layers
        else:
            self.drop_layers = None

    def forward(self, node_code, node_attr, edge_attr, adj, return_edge_code=False):

        # x = self.node_encoder(node_attr)
        x = node_code

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



class SEIRAL_GNN_4Input(nn.Module):

    def __init__(self, args):
        super(SEIRAL_GNN_4Input, self).__init__()

        nhid = args.len_hidden
        n_node_attr = args.len_node_attr
        n_node_code = nhid
        self.gnn_encoder_4input1 = GIN_4input(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=args.gnn_layers, use_gpu=False,
                               dropout=args.dropout)
        self.gnn_encoder_4input2 = GIN_4input(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                                              n_edge_attr=args.len_edge_attr, n_layers=args.gnn_layers, use_gpu=False,
                                              dropout=args.dropout)
        # -------------------------------------------------------------#
        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_attr, n_node_code),
            nn.LeakyReLU(0.1),
        )
        # -------------------------------------------------------------#

        self.lin1 = torch.nn.Linear(3*nhid, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr1, edge_attr2, adj = input

        node_code = self.node_encoder(node_attr)
        x = self.gnn_encoder_4input1(node_code, node_attr, edge_attr1, adj)
        gnn_node_codes = self.gnn_encoder_4input2(x, node_attr, edge_attr2, adj)
        # gnn_node_codes1 = self.gnn_encoder_4input(node_code, node_attr, edge_attr1, adj)
        # gnn_node_codes2 = self.gnn_encoder_4input(node_code, node_attr, edge_attr2, adj)
        # gnn_node_codes = torch.cat([gnn_node_codes1, gnn_node_codes2], dim=2)

        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, 1, :], gnn_node_codes[:, 2, :]], 1)
        # gnn_code = torch.cat([gnn_node_codes[:, 0, :],gnn_node_codes[:, 1, :],gnn_node_codes[:, 2, :],gnn_node_codes[:, 3, :],
        #                       gnn_node_codes[:, 4, :],gnn_node_codes[:, 5, :],gnn_node_codes[:, 6, :]],1)


        x = self.lin1(gnn_code)
        x = self.lin2(x)
        pred = self.output(x)

        return torch.sigmoid(pred)