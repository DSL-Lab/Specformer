import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.ops.edge_softmax import edge_softmax
from dgl.nn.pytorch.glob import AvgPooling
from dgl import function as fn
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [B, N]
        # output: [B, N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)
        
        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

    
class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Conv, self).__init__()

        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )

        self.preffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, graph, x_feat, edge_attr, bases):
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.apply_edges(fn.copy_u('x', '_x'))
            xee = self.pre_ffn(graph.edata['_x'] + edge_attr) * bases
            graph.edata['v'] = xee
            graph.update_all(fn.copy_e('v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
            y = graph.ndata['aggr_e']
            y = self.preffn_dropout(y)
            x = x_feat + y
            y = self.ffn(x)
            y = self.ffn_dropout(y)
            x = x + y
            return x


class SpecformerZINC(nn.Module):

    def __init__(self, nclass, nlayer, hidden_dim=128, nheads=4, feat_dropout=0.1, trans_dropout=0.1, adj_dropout=0.1):
        super(SpecformerZINC, self).__init__()
        
        self.nlayer = nlayer
        self.nclass = nclass
        self.hidden_dim = hidden_dim
        self.nheads = nheads

        self.atom_encoder = nn.Embedding(40, hidden_dim)
        self.bond_encoder = nn.Embedding(10, hidden_dim, padding_idx=0)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(trans_dropout)
        self.ffn_dropout = nn.Dropout(trans_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, trans_dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.adj_dropout = nn.Dropout(adj_dropout)
        self.filter_encoder = nn.Sequential(
            nn.Linear(nheads + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.convs = nn.ModuleList([Conv(hidden_dim, feat_dropout) for _ in range(nlayer)])
        self.pool = AvgPooling()
        self.linear = nn.Linear(hidden_dim, nclass)
        
    def forward(self, e, u, g, length):

        # e: [B, N]        eigenvalues
        # u: [B, N, N]     eigenvectors
        # x: [B, N, d]     node features
        # f: [B, N, N, d]  edge features   
        # do not use u to generate edge_idx because of the existing of connected components

        B, N = e.size()
        ut = u.transpose(1, 2)

        node_feat = g.ndata['feat']
        edge_feat = g.edata['feat']

        eig_mask, edge_idx = self.length_to_mask(length)

        node_feat = self.atom_encoder(node_feat).squeeze(-2)
        edge_feat = self.bond_encoder(edge_feat).squeeze(-2)

        eig = self.eig_encoder(e)
        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=eig_mask, average_attn_weights=False)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig).transpose(2, 1)      # [B, m, N]
        diag_e = torch.diag_embed(new_e)               # [B, m, N, N]

        bases = [torch.diag_embed(torch.ones_like(e))]
        for i in range(self.nheads):
            filters = u @ diag_e[:, i, :, :] @ ut
            bases.append(filters)

        bases = torch.stack(bases, axis=-1)  # [B, N, N, H]
        bases = bases[edge_idx]
        bases = self.adj_dropout(self.filter_encoder(bases))

        for conv in self.convs:
            node_feat = conv(g, node_feat, edge_feat, bases)

        h = self.pool(g, node_feat)
        h = self.linear(h)

        return h, new_e, attn


    def length_to_mask(self, length):
        '''
        length: [B]
        return: [B, max_len].
        '''
        B = len(length)
        N = length.max().item()
        mask1d  = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)

        mask2d = torch.zeros(B, N, N, device=length.device)
        for i in range(B):
            mask2d[i, :length[i], :length[i]] = 1.0

        # mask1d for key_padding_mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention
        # mask2d for edge selection from padding
        return mask1d, mask2d.bool()

