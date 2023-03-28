import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, second_dim_len=None, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)

        if second_dim_len:
            # if specified, use 2-dimensional embedding
            # following https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
            d_model = self.d_model // 2
            for pos in range(max_seq_len):
                h = pos // second_dim_len # the index of path
                w = pos % second_dim_len # the index of device on the path
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(h / (10000 ** ((2 * i) / d_model)))
                    pe[pos, i + 1] = math.cos(h / (10000 ** ((2 * (i + 1)) / d_model)))
                for i in range(d_model, d_model * 2, 2):
                    pe[pos, i] = math.sin(w / (10000 ** ((2 * i) / d_model)))
                    pe[pos, i + 1] = math.cos(w / (10000 ** ((2 * (i + 1)) / d_model)))
        else:
            # otherwise, 1-dimensional encoding
            for pos in range(max_seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class LearnablePositionEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.pe = nn.Parameter(torch.rand((1, max_seq_len, d_model)))

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)