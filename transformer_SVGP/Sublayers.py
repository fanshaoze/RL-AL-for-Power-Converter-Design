import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import transformer_config


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None, a_k=None):
    scores = torch.matmul(q, k.transpose(-2, -1))
    if a_k is not None:
        scores += torch.matmul(q.unsqueeze(-2), a_k.transpose(1, 2)).squeeze()
    scores /= math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


def create_rel_pe(w, seq_len):
    """
    Following implementation in https://arxiv.org/pdf/1809.04281.pdf
    The procedure is illustrated in Figure 1

    :param w: parameters used for position encoding. definition is in https://arxiv.org/pdf/1803.02155.pdf
    """
    # used to index neighbors in the range of [-path_len, +path_len]
    w_len = w.size(0)
    # feature dim
    d_k = w.size(1)

    # padding length on both side of w
    pad_size = seq_len - (w_len + 1) // 2
    row_padded = torch.cat((w[-1, :].expand(pad_size, d_k), w, w[-1, :].expand(pad_size, d_k)), dim=0)
    num_rows, _ = row_padded.shape

    matrix = row_padded.unsqueeze(0).expand(seq_len, num_rows, d_k)
    # matrix: seq_len * padded row size  * feat dim
    # padded row size = w_len + 2 * pad_size

    # add one dummy element to the end of each row
    padded_matrix = F.pad(matrix, (0, 0, 0, 1))
    seq_len, num_cols, d_k = padded_matrix.shape
    # padded_matrix: seq_len * (padded row size + 1) * feat dim

    unrolled_matrix = padded_matrix.view(-1, d_k)
    padded_unrolled_matrix = F.pad(unrolled_matrix, (0, 0, 0, num_cols - 1 - seq_len))
    reshaped_matrix = padded_unrolled_matrix.view(-1, num_cols - 1, d_k)
    # reshaped_matrx: (-1) * padded row size * feat dim

    # return the top-right corner of the matrix
    return reshaped_matrix[:seq_len, -seq_len:]


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, encoding, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.encoding = encoding

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.w_k = nn.Parameter(torch.normal(mean=0., std=1., size=(2 * transformer_config.max_path_len + 1, self.d_k)))

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        seq_len = q.size(1)

        if self.encoding == 'relative':
            a_k = create_rel_pe(self.w_k, seq_len)
        else:
            a_k = None

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, a_k=a_k)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        # print(self.w_k[0, :])

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
