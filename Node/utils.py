import time
import math
import random
import numpy as np
import scipy as sp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(dataset, y, nclass, seed=0):
    
    if dataset == 'arxiv':
        dataset = DglNodePropPredDataset('ogbn-arxiv')
        split = dataset.get_idx_split()
        train, valid, test = split['train'], split['valid'], split['test']
        return train, valid, test

    elif dataset == 'penn':
        split = np.load('node_raw_data/fb100-Penn94-splits.npy', allow_pickle=True)[0]
        train, valid, test = split['train'], split['valid'], split['test']
        return train, valid, test

    else:
        y = y.cpu()

        percls_trn = int(round(0.6 * len(y) / nclass))
        val_lb = int(round(0.2 * len(y)))

        indices = []
        for i in range(nclass):
            index = (y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0), device=index.device)]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_index = rest_index[:val_lb]
        test_index = rest_index[val_lb:]

        return train_index, valid_index, test_index

