import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset
import torch.nn.functional as F
import random
random.seed(61)

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
np.random.seed(123)
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from UGtransformer.datasets.dataset_brca_pyg import CancerDataset,get_class_weight

########
def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_name):
    dataset_path = '../datasets/' + dataset_name + '/mRNA'
    dataset = CancerDataset(dataset_path)

    # 获取标签字典和特征字典
    label_dict = {}
    feat_dict = {}
    for data in dataset:
        if data.y.item() not in label_dict:
            label_dict[data.y.item()] = len(label_dict)
        for tag in data.x.argmax(dim=1).tolist():
            if tag not in feat_dict:
                feat_dict[tag] = len(feat_dict)

    #过采样权重矩阵
    #weights = [1, 0.072, 0.187, 0.5, 0.215]
    #sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    # 创建DataLoader
    #loader = DataLoader(dataset, batch_size=1, sampler=sampler)

    print('# classes: %d' % len(label_dict))

    print("# data: %d" % len(dataset))

    loader = DataLoader(dataset, batch_size=16)

    return loader, len(label_dict), len(feat_dict)

def separate_data(loader, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [data.y for data in loader]
    labels = torch.cat(labels, dim=0)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    dataset = loader.dataset

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)


    return train_dataset, test_dataset

def get_kfold_data(k, i, loader):
    dataset = loader.dataset.shuffle()


    fold_size = len(dataset) // k
    test_start = i * fold_size
    if i != k-1:
        test_end = (i+1) * fold_size
        X_test = dataset[test_start:test_end]
        X_train = ConcatDataset((dataset[0:test_start], dataset[test_end:]))
    else:
        X_test = dataset[test_start:]
        X_train = dataset[0:test_start]


    num_training = int(len(X_train) * 8 // 9)
    num_val = int(len(X_train) - num_training)

    #training_set, validation_set = random_split(X_train, [num_training, num_val])
    train_set = X_train
    test_set = X_test

    #计算每个类别采样权重
    train_w = get_class_weight(train_set,'BRCA')
    test_w = get_class_weight(test_set, "BRCA")

    train_sampler = WeightedRandomSampler(train_w, num_samples=len(train_set), replacement=True)
    test_sampler = WeightedRandomSampler(test_w, num_samples=len(test_set), replacement=True)

    train_loader = DataLoader(train_set, batch_size=16, drop_last=True,sampler=train_sampler)
    #val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, drop_last=True, sampler=test_sampler)


    #return training_set, validation_set, X_test
    return train_loader, test_loader

"""Get indexes of train and test sets"""
def separate_data_idx(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

"""Convert sparse matrix to tuple representation."""
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



if __name__ == '__main__':
    load_data('BRCA')