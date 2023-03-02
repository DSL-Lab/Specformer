from dataclass import *
from dgldataclass import DglGraphPropPredDataset, DglPCQM4Mv2Dataset, DglZincDataset
from pygdataclass import PygGraphPropPredDataset
import dgl
from dgl.data.utils import load_graphs, save_graphs, Subset
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, RandomSampler
from torch_geometric.data import InMemoryDataset, Data
from ogb.graphproppred import Evaluator
from torch_geometric.utils import to_dense_adj


class PCQM4Mv2Evaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4Mv2 dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'].reshape(-1), input_dict['y_true'].reshape(-1)

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)
        assert(len(y_true.shape) == 1)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}


class DynamicBatchSampler(Sampler):
    def __init__(self, sampler, num_nodes_list, batch_size=32, max_nodes=200, drop_last=False):

        super(DynamicBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.num_nodes_list = num_nodes_list
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.drop_last = drop_last

    def __iter__(self):

        batch = []
        total_nodes = 0
        memory = self.max_nodes * self.max_nodes * self.batch_size

        for idx in self.sampler:
            cur_nodes = self.num_nodes_list[idx]

            # beyond memory, truncate batch
            # squre for Transformer
            if total_nodes + cur_nodes ** 2 > memory:
                yield batch
                batch = [idx]
                total_nodes = cur_nodes ** 2
            else:
                batch.append(idx)
                total_nodes += cur_nodes ** 2

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                total_nodes = 0

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))

    graph_list = []
    length = []
    E = []
    U = []

    max_nodes = max([g.num_nodes() for g in graphs])

    for i, g in enumerate(graphs):
        num_nodes = g.num_nodes()

        e = g.ndata['e']
        u = g.ndata['u']

        pad_e = e.new_zeros([max_nodes])
        pad_e[:num_nodes] = e

        pad_u = u.new_zeros([max_nodes, max_nodes])
        pad_u[:num_nodes, :num_nodes] = u

        E.append(pad_e)
        U.append(pad_u)
        graph_list.append(g)
        length.append(num_nodes)

    E = torch.stack(E, 0)
    U = torch.stack(U, 0)
    length = torch.LongTensor(length)
    batched_graph = dgl.batch(graphs, ndata=['feat'], edata=['feat'])

    if isinstance(labels[0], torch.Tensor):
        return E, U, batched_graph, length, torch.stack(labels)
    else:
        return E, U, batched_graph, length, labels


def collate_pyg(graphs):
    graph_list = []
    length = []
    E = []
    U = []
    Y = []

    max_nodes = max([g.num_nodes for g in graphs])

    for g in graphs:
        num_nodes = g.num_nodes

        e = g.e
        u = g.u.view(num_nodes, num_nodes)
        
        pad_e = e.new_zeros([max_nodes])
        pad_e[:num_nodes] = e

        pad_u = u.new_zeros([max_nodes, max_nodes])
        pad_u[:num_nodes, :num_nodes] = u

        E.append(pad_e)
        U.append(pad_u)
        Y.append(g.y.view(-1))
        length.append(num_nodes)
        
        fully_connected = torch.ones([num_nodes, num_nodes], dtype=torch.float).nonzero(as_tuple=True)
        dgl_graph = dgl.graph(fully_connected, num_nodes = num_nodes)
        edge_attr_dense = to_dense_adj(g.edge_index, edge_attr=g.edge_attr, 
                                       max_num_nodes=num_nodes).squeeze(0).view(-1, g.edge_attr.shape[-1])
        dgl_graph.ndata['feat'] = torch.zeros(num_nodes, 1)
        dgl_graph.edata['feat'] = edge_attr_dense
        graph_list.append(dgl_graph)

    E = torch.stack(E, 0)
    U = torch.stack(U, 0)
    Y = torch.stack(Y, 0)
    length = torch.LongTensor(length)
    batched_graph = dgl.batch(graph_list, ndata=['feat'], edata=['feat'])

    return E, U, batched_graph, length, Y


def collate_pad(batch):
    E = []
    U = []
    X = []
    F = []
    Y = []
    
    max_nodes = min(max([data.num_nodes for data in batch]), 128)

    for data in batch:
        length = data.num_nodes
        e = data.e
        u = data.u.view(length, length)
        x = data.x
        f = data.edge_attr

        if length > max_nodes:
            src, dst = data.edge_index
            A = torch.zeros([length, length], dtype=torch.float)
            A[src, dst] = 1.0
            A = A[:max_nodes, :max_nodes]
            deg = torch.sum(A, axis=0).squeeze()
            deg = torch.clamp(deg, min=1.0) ** -0.5
            D = torch.diag(deg)
            A_ = D @ A @ D

            pad_e, pad_u = torch.linalg.eigh(A_)
            pad_x = x[:max_nodes, :] + 1

            fdim = f.size(-1)
            pad_f = torch.zeros([length, length, fdim], dtype=torch.long)
            pad_f[src, dst] = f + 1
            pad_f = pad_f[:max_nodes, :max_nodes] 
        else:
            pad_e = e.new_zeros([max_nodes])
            pad_e[:length] = e

            pad_u = u.new_zeros([max_nodes, max_nodes])
            pad_u[:length, :length] = u

            xdim = x.size(-1)
            pad_x = x.new_zeros([max_nodes, xdim])
            pad_x[:length, :] = x + 1

            fdim = f.size(-1)
            src, dst = data.edge_index
            pad_f = f.new_zeros([max_nodes, max_nodes, fdim])
            pad_f[src, dst, :] = f + 1 

        E.append(pad_e)
        U.append(pad_u)
        X.append(pad_x)
        F.append(pad_f)
        Y.append(data.y.squeeze())

    return torch.stack(E, 0), torch.stack(U, 0), torch.stack(X, 0), torch.stack(F, 0), torch.stack(Y, 0)


def get_dataset(dataset_name='abaaba'):

    if dataset_name == 'zinc':
        data_info = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': PCQM4Mv2Evaluator(),
            'train_dataset': DglZincDataset('dataset/zinc', subset=True, split='train'),
            'valid_dataset': DglZincDataset('dataset/zinc', subset=True, split='val'),
            'test_dataset':  DglZincDataset('dataset/zinc', subset=True, split='test'),
        }
    elif dataset_name == 'pcqm':
        dataset = DglPCQM4Mv2Dataset()
        split_idx = dataset.get_idx_split()
        idx = split_idx['train']
        rand_idx = torch.randperm(idx.size(0))
        train_idx = idx[rand_idx[150000:]]
        valid_idx = idx[rand_idx[:150000]]
        test_idx = split_idx['valid']

        data_info = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': PCQM4Mv2Evaluator(),
            'train_dataset': dataset[train_idx],
            'valid_dataset': dataset[valid_idx],
            'test_dataset':  dataset[test_idx],
        }
    elif dataset_name == 'pcqms':
        train_g, train_dict = load_graphs('dataset/pcqm_subset_train.pt')
        valid_g, valid_dict = load_graphs('dataset/pcqm_subset_valid.pt')
        test_g,  test_dict  = load_graphs('dataset/pcqm_subset_test.pt')

        data_info = {
            'num_class': 1,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': PCQM4Mv2Evaluator(),
            'train_dataset': list(zip(train_g, train_dict['labels'])),
            'valid_dataset': list(zip(valid_g, valid_dict['labels'])),
            'test_dataset':  list(zip(test_g, test_dict['labels'])),
        }
    elif dataset_name == 'hiv':
        dataset = DglGraphPropPredDataset('ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        data_info = {
            'num_class': 1,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': Evaluator('ogbg-molhiv'),
            'train_dataset': dataset[split_idx['train']],
            'valid_dataset': dataset[split_idx['valid']],
            'test_dataset':  dataset[split_idx['test']],
        }
    elif dataset_name == 'pcba':
        dataset = DglGraphPropPredDataset(name = 'ogbg-molpcba')
        split_idx = dataset.get_idx_split()
        data_info = {
            'num_class': 128,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'ap',
            'metric_mode': 'max',
            'evaluator': Evaluator('ogbg-molpcba'),
            'train_dataset': dataset[split_idx['train']],
            'valid_dataset': dataset[split_idx['valid']],
            'test_dataset':  dataset[split_idx['test']],
        }
    elif dataset_name == 'ppa':
        dataset = PygGraphPropPredDataset(name = 'ogbg-ppa')
        split_idx = dataset.get_idx_split()
        data_info = {
            'num_class': 37,
            'loss_fn':  F.cross_entropy,
            'metric': 'acc',
            'metric_mode': 'max',
            'evaluator': Evaluator('ogbg-ppa'),
            'train_dataset': dataset[split_idx['train']],
            'valid_dataset': dataset[split_idx['valid']],
            'test_dataset':  dataset[split_idx['test']],
        }
    else:
        raise NotImplementedError

    return data_info

