import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from dgl.nn.pytorch.glob import AvgPooling
from dgl import function as fn
from dgl.ops.edge_softmax import edge_softmax
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for _, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for _, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)  # for padding
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


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


class SpecformerSmall(nn.Module):

    def __init__(self, nclass, nlayer, hidden_dim=128, nheads=4, feat_dropout=0.1, trans_dropout=0.1, adj_dropout=0.1):
        super(SpecformerSmall, self).__init__()
        
        print('small model')
        self.nlayer = nlayer
        self.nclass = nclass
        self.hidden_dim = hidden_dim
        self.nheads = nheads

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

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
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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

        B, N = e.size()
        ut = u.transpose(1, 2)

        node_feat = g.ndata['feat']
        edge_feat = g.edata['feat']

        # do not use u to generate edge_idx because of the connected components
        e_mask, edge_idx = self.length_to_mask(length)

        node_feat = self.atom_encoder(node_feat)
        edge_feat = self.bond_encoder(edge_feat)
        eig = self.eig_encoder(e)

        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=e_mask)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig).transpose(2, 1)      # [B, m, N]
        diag_e = torch.diag_embed(new_e)               # [B, m, N, N]

        identity = torch.diag_embed(torch.ones_like(e))
        bases = [identity]
        for i in range(self.nheads):
            filters = u @ diag_e[:, i, :, :] @ ut
            bases.append(filters)

        bases = torch.stack(bases, axis=-1)  # [B, N, N, H]
        bases = bases[edge_idx]
        bases = self.adj_dropout(self.filter_encoder(bases))
        bases = edge_softmax(g, bases)

        for conv in self.convs:
            node_feat = conv(g, node_feat, edge_feat, bases)

        h = self.pool(g, node_feat)
        h = self.linear(h)

        return h


    def length_to_mask(self, length):
        '''
        length: [B]
        return: [B, max_len].
        '''
        B = len(length)
        N = length.max().item()
        mask1d = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)
        mask2d = (~mask1d).float().unsqueeze(2) @ (~mask1d).float().unsqueeze(1)
        mask2d = mask2d.bool()

        # Example
        # length=[1, 2, 3], B=3, N=3,

        # mask1d for key_padding_mask of MultiheadAttention [B, N]
        # [False, True,  True ]
        # [False, False, True ]
        # [False, False, False]

        # mask2d for edge indexing [B, N, N]
        # [[1, 0, 0], | [1, 1, 0], | [1, 1, 1],
        #  [0, 0, 0], | [1, 1, 0], | [1, 1, 1],
        #  [0, 0, 0], | [0, 0, 0], | [1, 1, 1],]

        return mask1d, mask2d


    '''
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
    '''
