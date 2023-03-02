import time
import math
import copy
import wandb
import argparse
import datetime
import random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import json5
from easydict import EasyDict

from ema_pytorch import EMA
from zinc_model import SpecformerZINC
from large_model import SpecformerLarge
from medium_model import SpecformerMedium
from small_model import SpecformerSmall
from get_dataset import DynamicBatchSampler, RandomSampler, collate_pad, collate_dgl, get_dataset


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_config_from_json(json_file):
    with open('config/' + json_file + '.json', 'r') as config_file:
        config_dict = json5.load(config_file)
    config = EasyDict(config_dict)

    return config


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(dataset, model, device, dataloader, loss_fn, optimizer, wandb=None, wandb_item=None):
    model.train()

    for i, data in enumerate(dataloader):
        e, u, g, length, y = data
        e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)

        logits = model(e, u, g, length)
        optimizer.zero_grad()

        y_idx = y == y
        loss = loss_fn(logits.to(torch.float32)[y_idx], y.to(torch.float32)[y_idx])

        loss.backward()
        optimizer.step()

        if wandb:
            wandb.log({wandb_item: loss.item()})


def eval_epoch(dataset, model, device, dataloader, evaluator, metric):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            e, u, g, length, y = data
            e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)

            logits = model(e, u, g, length)

            y_true.append(y.view(logits.shape).detach().cpu())
            y_pred.append(logits.detach().cpu())
        
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})[metric]


def main_worker(args):
    seed_everything(args.seed)
    rank = 'cuda:{}'.format(args.cuda)
    print(args)

    datainfo = get_dataset(args.dataset)
    nclass = datainfo['num_class']
    loss_fn = datainfo['loss_fn']
    evaluator = datainfo['evaluator']
    train = datainfo['train_dataset']
    valid = datainfo['valid_dataset']
    test  = datainfo['test_dataset']
    metric = datainfo['metric']
    metric_mode = datainfo['metric_mode']

    # dataloader
    '''
    train_batch_sampler = DynamicBatchSampler(RandomSampler(train), [data.num_nodes for data in train],
                                              batch_size=32, max_nodes=50, drop_last=False)
    valid_batch_sampler = DynamicBatchSampler(RandomSampler(valid), [data.num_nodes for data in valid],
                                              batch_size=32, max_nodes=50, drop_last=False)
    test_batch_sampler  = DynamicBatchSampler(RandomSampler(test),  [data.num_nodes for data in test],
                                              batch_size=32, max_nodes=50, drop_last=False)
    train_dataloader = DataLoader(train, batch_sampler=train_batch_sampler, collate_fn=collate_pad)
    valid_dataloader = DataLoader(valid, batch_sampler=valid_batch_sampler, collate_fn=collate_pad)
    test_dataloader  = DataLoader(test,  batch_sampler=test_batch_sampler,  collate_fn=collate_pad)
    '''

    train_dataloader = DataLoader(train, batch_size = args.batch_size, num_workers=4, collate_fn=collate_dgl, shuffle = True)
    valid_dataloader = DataLoader(valid, batch_size = args.batch_size // 2, num_workers=4, collate_fn=collate_dgl, shuffle = False)
    test_dataloader  = DataLoader(test,  batch_size = args.batch_size // 2, num_workers=4, collate_fn=collate_dgl, shuffle = False)

    if args.dataset == 'zinc':
        print('zinc')
        model = SpecformerZINC(nclass, args.nlayer, args.hidden_dim, args.nheads,
                               args.feat_dropout, args.trans_dropout, args.adj_dropout).to(rank)

    elif args.dataset == 'pcqm' or args.dataset == 'pcqms':
        print('pcqm')
        model = SpecformerLarge(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                 args.feat_dropout, args.trans_dropout, args.adj_dropout).to(rank)
        print('init')
        model.apply(init_params)

    elif args.dataset == 'pcba':
        print('pcba')
        model = SpecformerMedium(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                 args.feat_dropout, args.trans_dropout, args.adj_dropout).to(rank)
        model.apply(init_params)

    else:
        print('hiv')
        model = SpecformerSmall(nclass, args.nlayer, args.hidden_dim, args.nheads,
                                args.feat_dropout, args.trans_dropout, args.adj_dropout).to(rank)

    print(count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # warm_up + cosine weight decay
    lr_plan = lambda cur_epoch: (cur_epoch+1) / args.warm_up_epoch if cur_epoch < args.warm_up_epoch else \
              (0.5 * (1.0 + math.cos(math.pi * (cur_epoch - args.warm_up_epoch) / (args.epochs - args.warm_up_epoch))))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_plan)

    results = []
    for epoch in range(args.epochs):

        train_epoch(args.dataset, model, rank, train_dataloader, loss_fn, optimizer, wandb=None, wandb_item='loss')
        scheduler.step()

        torch.save(model.state_dict(), 'checkpoint/{}_{}.pth'.format(args.project_name, epoch))

        if epoch % 1 == 0:

            val_res = eval_epoch(args.dataset, model, rank, valid_dataloader, evaluator, metric)
            test_res = eval_epoch(args.dataset, model, rank, test_dataloader, evaluator, metric)

            results.append([val_res, test_res])

            if metric_mode == 'min':
                best_res = sorted(results, key = lambda x: x[0], reverse=False)[0][1]
            else:
                best_res = sorted(results, key = lambda x: x[0], reverse=True)[0][1]

            print(epoch, 'valid: {:.4f}'.format(val_res), 'test: {:.4f}'.format(test_res), 'best: {:.4f}'.format(best_res))

            # wandb.log({'val': val_res, 'test': test_res})

    torch.save(model.state_dict(), 'checkpoint/{}.pth'.format(args.project_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='zinc')

    args = parser.parse_args()
    args.project_name = datetime.datetime.now().strftime('%m-%d-%X')

    config = get_config_from_json(args.dataset)

    for key in config.keys():
        setattr(args, key, config[key])

    main_worker(args)

