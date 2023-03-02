import pickle
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
import dgl
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw, read_binary_graph_raw, read_binary_heterograph_raw
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, remove_isolated_nodes
from torch_geometric.data import InMemoryDataset


def read_graph_dgl(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = [], binary=False):

    if binary:
        # npz
        graph_list = read_binary_graph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
        
    dgl_graph_list = []

    print('Converting graphs into DGL objects...')
    
    for graph in tqdm(graph_list):

        src, dst = torch.from_numpy(graph['edge_index'])
        num_nodes = graph['num_nodes']

        if num_nodes == 1:  # some graphs have one node
            A_ = torch.tensor(1.).view(1, 1)
        else:
            A = torch.zeros([num_nodes, num_nodes], dtype=torch.float)
            A[src, dst] = 1.0
            for i in range(num_nodes):
                A[i, i] = 1.0
            deg = torch.sum(A, axis=0).squeeze() ** -0.5
            D = torch.diag(deg)
            A_ = D @ A @ D
        e, u = torch.linalg.eigh(A_)

        fully_connected = torch.ones([num_nodes, num_nodes], dtype=torch.float).nonzero(as_tuple=True)
        g = dgl.graph(fully_connected, num_nodes = num_nodes)

        g.ndata['e'] = e
        g.ndata['u'] = u

        if graph['node_feat'] is not None:
            g.ndata['feat'] = torch.from_numpy(graph['node_feat'])

        if graph['edge_feat'] is not None:
            edge_idx = torch.stack([src, dst], dim=0)
            edge_attr = torch.from_numpy(graph['edge_feat']) + 1  # for padding

            if len(edge_attr.shape) == 1:
                edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr.unsqueeze(-1)).squeeze(0).squeeze(-1).view(-1)
            else:
                if edge_attr.size(0) == 0:
                    edge_attr_dense = torch.zeros([num_nodes ** 2, edge_attr.size(1)]).long()  # for graphs without edge
                else:
                    edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr, max_num_nodes=num_nodes).squeeze(0).view(-1, edge_attr.shape[-1])

            g.edata['feat'] = edge_attr_dense

        dgl_graph_list.append(g)

    return dgl_graph_list


class DglGraphPropPredDataset(object):
    '''Adapted from https://docs.dgl.ai/en/latest/_modules/dgl/data/chem/csv_dataset.html#CSVDataset'''
    def __init__(self, name, root = 'dataset', meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name ## original name, e.g., ogbg-molhiv

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_dgl')):
                self.dir_name = self.dir_name + '_dgl'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.num_classes = self.meta_info['num classes']
        self.binary = self.meta_info['binary'] == 'True'

        super(DglGraphPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if self.task_type == 'subtoken prediction':
            target_sequence_file_path = osp.join(processed_dir, 'target_sequence')

        if os.path.exists(pre_processed_file_path):

            if self.task_type == 'subtoken prediction':
                self.graphs, _ = load_graphs(pre_processed_file_path)
                self.labels = torch.load(target_sequence_file_path)

            else:
                self.graphs, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']

        else:
            ### check download
            if self.binary:
                # npz format
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'data.npz'))
            else:
                # csv file
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'edge.csv.gz'))

            ### download
            if not has_necessary_file:
                url = self.meta_info['url']
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(osp.join(self.original_root, self.download_name), self.root)
                else:
                    print('Stop download.')
                    exit(-1)

            ### preprocess
            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(',')

            graphs = read_graph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)

            if self.task_type == 'subtoken prediction':
                # the downloaded labels are initially joined by ' '
                labels_joined = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
                # need to split each element into subtokens
                labels = [str(labels_joined[i][0]).split(' ') for i in range(len(labels_joined))]

                print('Saving...')
                save_graphs(pre_processed_file_path, graphs)
                torch.save(labels, target_sequence_file_path)

                ### load preprocessed files
                self.graphs, _ = load_graphs(pre_processed_file_path)
                self.labels = torch.load(target_sequence_file_path)

            else:
                if self.binary:
                    labels = np.load(osp.join(raw_dir, 'graph-label.npz'))['graph_label']
                else:
                    labels = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

                has_nan = np.isnan(labels).any()

                if 'classification' in self.task_type:
                    if has_nan:
                        labels = torch.from_numpy(labels).to(torch.float32)
                    else:
                        labels = torch.from_numpy(labels).to(torch.long)
                else:
                    labels = torch.from_numpy(labels).to(torch.float32)


                print('Saving...')
                save_graphs(pre_processed_file_path, graphs, labels={'labels': labels})

                ### load preprocessed files
                self.graphs, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']


    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class DglPCQM4Mv2Dataset(object):
    def __init__(self, root = 'dataset', smiles2graph = smiles2graph):
        '''
        DGL PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(DglPCQM4Mv2Dataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
        self.prepare_graph()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):        
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']
        else:
            # if pre-processed file does not exist
            
            if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
                # if the raw file does not exist, then download it.
                self.download()

            data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
            smiles_list = data_df['smiles']
            homolumogap_list = data_df['homolumogap']

            print('Converting SMILES strings into graphs...')
            self.graphs = []
            self.labels = []
            for i in tqdm(range(len(smiles_list))):

                smiles = smiles_list[i]
                homolumogap = homolumogap_list[i]
                graph = self.smiles2graph(smiles)
                
                assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert(len(graph['node_feat']) == graph['num_nodes'])

                src, dst = torch.from_numpy(graph['edge_index'])
                num_nodes = graph['num_nodes']

                if num_nodes == 1:  # some graphs have one node
                    A_ = torch.tensor(1.).view(1, 1)
                else:
                    A = torch.zeros([num_nodes, num_nodes], dtype=torch.float)
                    A[src, dst] = 1.0
                    for i in range(num_nodes):
                        A[i, i] = 1.0
                    deg = torch.sum(A, axis=0).squeeze() ** -0.5
                    D = torch.diag(deg)
                    A_ = D @ A @ D
                e, u = torch.linalg.eigh(A_)

                fully_connected = torch.ones([num_nodes, num_nodes], dtype=torch.float).nonzero(as_tuple=True)
                g = dgl.graph(fully_connected, num_nodes = num_nodes)

                g.ndata['e'] = e
                g.ndata['u'] = u

                if graph['node_feat'] is not None:
                    g.ndata['feat'] = torch.from_numpy(graph['node_feat']).long()

                if graph['edge_feat'] is not None:
                    edge_idx = torch.stack([src, dst], dim=0)
                    edge_attr = torch.from_numpy(graph['edge_feat']).long() + 1  # for padding

                    if len(edge_attr.shape) == 1:
                        edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr.unsqueeze(-1)).squeeze(0).squeeze(-1).view(-1)
                    else:
                        if edge_attr.size(0) == 0:   # for graphs without edge
                            edge_attr_dense = torch.zeros([num_nodes ** 2, edge_attr.size(1)]).long()
                        else:
                            edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr, max_num_nodes=num_nodes).squeeze(0).view(-1, edge_attr.shape[-1])

                g.edata['feat'] = edge_attr_dense 

                self.graphs.append(g)
                self.labels.append(homolumogap)

            self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # double-check prediction target
            split_dict = self.get_idx_split()
            assert(all([not torch.isnan(self.labels[i]) for i in split_dict['train']]))
            assert(all([not torch.isnan(self.labels[i]) for i in split_dict['valid']]))
            assert(all([torch.isnan(self.labels[i]) for i in split_dict['test-dev']]))
            assert(all([torch.isnan(self.labels[i]) for i in split_dict['test-challenge']]))

            print('Saving...')
            save_graphs(pre_processed_file_path, self.graphs, labels={'labels': self.labels})


    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.folder, 'split_dict.pt')))
        return split_dict

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class DglZincDataset(InMemoryDataset):

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'dgl_{split}.pt')
        print(path)
        self.graphs, label_dict = load_graphs(path)
        self.labels = label_dict['labels']

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['dgl_train.pt', 'dgl_val.pt', 'dgl_test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)
           
            indices = range(len(mols))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            graphs = []
            labels = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_idx = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_idx[0], edge_idx[1]].to(torch.long) + 1  # for padding

                src, dst = edge_idx
                num_nodes = x.size(0)

                if num_nodes == 1:  # some graphs have one node
                    A_ = torch.tensor(1.).view(1, 1)
                else:
                    A = torch.zeros([num_nodes, num_nodes], dtype=torch.float)
                    A[src, dst] = 1.0
                    for i in range(num_nodes):
                        A[i, i] = 1.0
                    deg = torch.sum(A, axis=0).squeeze() ** -0.5
                    D = torch.diag(deg)
                    A_ = D @ A @ D
                e, u = torch.linalg.eigh(A_)

                fully_connected = torch.ones([num_nodes, num_nodes], dtype=torch.float).nonzero(as_tuple=True)
                g = dgl.graph(fully_connected, num_nodes = num_nodes)

                g.ndata['e'] = e
                g.ndata['u'] = u

                g.ndata['feat'] = x
                g.edata['feat'] = to_dense_adj(edge_idx, edge_attr=edge_attr.unsqueeze(-1)).squeeze(0).squeeze(-1).view(-1)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                graphs.append(g)
                labels.append(y)

                pbar.update(1)

            pbar.close()

            labels = torch.tensor(labels, dtype=torch.float32)
            save_graphs(osp.join(self.processed_dir, f'dgl_{split}.pt'), graphs, labels={'labels': labels})

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

