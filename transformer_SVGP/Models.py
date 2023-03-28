import pprint

import torch
import torch.nn as nn

import transformer_config
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder, LearnablePositionEncoder
from Sublayers import FeedForward, MultiHeadAttention, Norm
import copy
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

Constants_PAD = 0


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N_layers, heads, encoding, dropout):
        super().__init__()
        self.N_layers = N_layers
        # self.embed = Embedder(vocab_size, d_model)
        self.encoding = encoding
        if encoding == 'absolute_2d':
            self.pe = PositionalEncoder(d_model, second_dim_len=transformer_config.max_path_len, dropout=dropout)
        elif encoding == 'absolute':
            self.pe = PositionalEncoder(d_model, dropout=dropout)
        elif encoding == 'learnable':
            self.pe = LearnablePositionEncoder(d_model, dropout=dropout)
        # otherwise, no position encoding (pe) is used

        self.layers = get_clones(EncoderLayer(d_model, heads, encoding, dropout), N_layers)
        self.norm = Norm(d_model)

    def forward(self, x, mask=None):
        # x = self.embed(src)
        if self.encoding in ['absolute', 'absolute_2d', 'learnable']:
            x = self.pe(x)
        # x = src
        for i in range(self.N_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N_layers, heads, dropout):
        super().__init__()
        self.N_layers = N_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N_layers)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N_layers):
            x = self.layers[i](x, e_outputs, src_mask=None, trg_mask=trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            N,
            heads,
            dropout,
            encoding,
            duty_encoding,
            mlp_layers,
            output_size=1,
            device=torch.device('cuda')
    ):
        super().__init__()

        # other configs
        self.bidirectional = transformer_config.bidirectional
        self.average_pooling = transformer_config.average_pooling

        self.use_lstm = (encoding == 'lstm')
        self.use_hierarchical_transformer = (encoding == 'hierarchical_transformer')
        self.embedding = Embedder(vocab_size, d_model)

        self.duty_encoding = duty_encoding

        if self.duty_encoding == 'path':
            self.duty_encoder = nn.Sequential(nn.Linear(1, int(d_model / 2)), nn.ReLU())

        if self.use_lstm:
            self.lstm_output_dim = d_model
            self.lstm = nn.LSTM(d_model, self.lstm_output_dim, bidirectional=self.bidirectional, batch_first=True).to(
                device)

        transformer_feat_dim = d_model * 2 if self.use_lstm and self.bidirectional else d_model
        if self.duty_encoding == 'path':
            transformer_feat_dim += int(d_model / 2)

        self.encoder = Encoder(d_model=transformer_feat_dim,
                               N_layers=N,
                               heads=heads,
                               encoding=encoding,
                               dropout=dropout)

        if self.use_hierarchical_transformer:
            self.path_encoder = Encoder(d_model=d_model,
                                        N_layers=N,
                                        heads=heads,
                                        encoding='absolute',  # the order of components matters as we encode a path
                                        dropout=dropout)

        # adding more layers here doesn't seem to help in performance
        mlp_layers = [transformer_feat_dim] + mlp_layers
        out1_list = []
        for in_size, out_size in zip(mlp_layers[:-1], mlp_layers[1:]):
            out1_list.append(nn.Linear(in_size, out_size))
            out1_list.append(nn.Dropout(dropout))
            out1_list.append(nn.ReLU())

        self.out1 = nn.Sequential(*out1_list)

        if self.duty_encoding == 'mlp':
            # leave space for duty input
            self.out2 = nn.Sequential(nn.Linear(mlp_layers[-1] + 1, output_size),
                                      nn.Sigmoid())
        else:
            self.out2 = nn.Sequential(nn.Linear(mlp_layers[-1], output_size),
                                      nn.Sigmoid())

        self.d_model = d_model
        self.device = device

    def forward(self, x, duty, padding_mask):
        """
        :param x: (batch size) x (maximum path num in a topology) x (maximum device num in a path)
            default values are 512 x 10 x 8
        :param padding_mask: (batch size) x (maximum path num in a topology) x (maximum device num in a path)
            it masks devices that are present
        """
        input_size = x.size()
        x = self.embedding(x)
        # x: (batch size) x (maximum path num in a topology) x (maximum device num in a path) x (embedding dim)

        if self.use_lstm:
            # prepare x for lstm
            # flatten all path indices
            x = x.view(-1, x.size(2), x.size(3))
            # x: (num of paths in all topologies) x (maximum device num in a path) x (d_model)

            # lengths of all paths in topos of this batch
            path_lengths = padding_mask.view(-1, padding_mask.size(2)).sum(dim=1)
            # indices of non-empty paths
            pack_indices = path_lengths.nonzero().flatten()

            # we want to let lstm process data in batch, however, different paths have different lengths,
            # and we don't want to let lstm process <PAD>
            # a standard way is to create a "padded sequence" that stores paths of different lengths
            x = torch.nn.utils.rnn.pack_padded_sequence(x[pack_indices, :, :], path_lengths[pack_indices].cpu(),
                                                        batch_first=True, enforce_sorted=False)
            out, (hidden, cell) = self.lstm(x)

            lstm_output_dim = 2 * self.lstm_output_dim if self.bidirectional else self.lstm_output_dim
            lstm_out = torch.zeros((path_lengths.size(0), lstm_output_dim)).to(self.device)
            if self.average_pooling:
                # if average pooling, get the mean of the states of all paths (excluding the padding)
                padded_out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
                out_sums = torch.sum(padded_out, dim=1)
                out_means = out_sums / path_lengths[pack_indices].view(-1, 1)
                lstm_out[pack_indices, :] = out_means
            else:
                # use the hidden state of the last element in the path
                lstm_out[pack_indices, :] = hidden.squeeze()

            # now path embeddings are generated (using LSTM)
            # reshape paths back according to topos
            x = lstm_out.view(input_size[0], input_size[1], lstm_output_dim)
            # x: (batch size) x (maximum path num in a topology) x (lstm output dim)

            # compute which path to mask based on device-level mask
            padding_mask = padding_mask.sum(axis=2, dtype=bool).unsqueeze(2)
            # padding_mask: (batch size) x (maximum path num in a topology) x 1
        elif self.use_hierarchical_transformer:
            x = x.view(-1, x.size(2), x.size(3))
            # x: (num of paths in all topologies) x (maximum device num in a path) x (d_model)
            path_padding_mask = padding_mask.view(x.size(0), x.size(1), 1)
            # path_padding_mask: (num of paths in all topologies) x (maximum device num in a path) x 1

            x = self.path_encoder(x, path_padding_mask)
            # x: (num of paths in all topologies) x (maximum device num in a path) x (d_model)
            x = x.view(input_size[0], input_size[1], input_size[2], self.d_model)
            # x: (batch size) x (maximum path num in a topology) x (maximum device num in a path) x (d_model)

            # aggregate over all components in a path
            x = torch.mean(x, dim=-2)
            # x: (batch size) x (maximum path num in a topology) x (d_model)

            # compute which path to mask
            padding_mask = padding_mask.sum(axis=2, dtype=bool).unsqueeze(2)
        else:
            # flatten all paths in a topo
            x = x.view(x.size(0), -1, x.size(3))
            padding_mask = padding_mask.view(padding_mask.size(0), -1).unsqueeze(2)

        if self.duty_encoding == 'path':
            # add duty here
            encoded_duty = self.duty_encoder(duty)
            # concatenate encoded duty to each path embedding
            x = torch.cat([x, encoded_duty.unsqueeze(1).expand(-1, x.size(1), -1)], dim=2)

        # x:            (batch size) x (maximum path num in a topology) x (d_model)
        # padding_mask: (batch size) x (maximum path num in a topology) x 1
        x = self.encoder(x, padding_mask)
        # average embeddings for all paths in a topology, to obtain the embedding for the whole topology
        x_1 = torch.mean(x, dim=1)

        x = self.out1(x_1)
        if self.duty_encoding == 'mlp':
            # add duty to the final layer
            # and make predictions to be between 0 and 1
            x = self.out2(torch.hstack([x, duty]))
        else:
            x = self.out2(x)

        return x_1, x


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_model(opt=None, pretrained_model=None, load_weights=False):
    pretrained_model = pretrained_model or opt.pretrained_model

    if load_weights:
        checkpoint = torch.load(pretrained_model + '.chkpt')
        model_opt = checkpoint['settings']

        pprint.pprint(vars(model_opt))

        model = Transformer(
            vocab_size=model_opt.vocab_size,
            d_model=model_opt.d_model,
            N=model_opt.n_layers,
            heads=model_opt.n_heads,
            dropout=model_opt.dropout,
            encoding=model_opt.encoding,
            duty_encoding=model_opt.duty_encoding,
            # eff_vout objective needs two outputs (two scalars), eff and vout
            # other targets needs one output
            mlp_layers=model_opt.mlp_layers,
            # output_size=2 if opt.target == 'eff_vout' else 1,
            output_size=1,
            device=model_opt.device
        )

        model.load_state_dict(checkpoint['model'])

        print('[Info] Trained model state loaded from: ', pretrained_model)


    else:
        assert opt.d_model % opt.n_heads == 0

        assert opt.dropout < 1

        model = Transformer(
            vocab_size=opt.vocab_size,
            d_model=opt.d_model,
            N=opt.n_layers,
            heads=opt.n_heads,
            dropout=opt.dropout,
            encoding=opt.encoding,
            duty_encoding=opt.duty_encoding,
            mlp_layers=opt.mlp_layers,
            output_size=2 if opt.target == 'eff_vout' else 1,
            device=opt.device
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    return np_mask


def create_masks(trg):
    # src_mask = (src != Constants_PAD.unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != Constants_PAD).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(trg_mask.device)

        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None

    return trg_mask
