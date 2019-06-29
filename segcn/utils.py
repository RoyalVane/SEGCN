import numpy as np
import scipy.sparse as sp
import torch
import random
import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
#np.set_printoptions(threshold=np.inf)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


# label: one_hot matrix 
# total: #all sample
# number: #select ? in each class
# classes: #class
def select_sample(labels, total, number_train = 20, number_val = 500, number_test = 1000, classes = 7): 
    idx_train = []
    idx_val = []
    idx_test = []
    all_set = set(range(total))
    for c in range(classes):
        pure_idx = [i for i in range(total) if labels[i][c] == 1]
        idx_train += random.sample(pure_idx, number_train)
    rest = list(all_set - set(idx_train))
    idx_val_test = random.sample(rest, number_val + number_test)
    idx_val = idx_val_test[:number_val]
    idx_test = idx_val_test[number_val:number_val + number_test]
    
    return idx_train, idx_val, idx_test


def select_sample2(labels, total, test_set, number_train = 20, number_val = 500, classes = 7): 
    idx_train = []
    idx_val = []
    idx_test = []
    train_val_set = set(range(total)) - test_set
    for c in range(classes):
        pure_idx = [i for i in train_val_set if labels[i][c] == 1]
        idx_train += random.sample(pure_idx, number_train)
    idx_val = random.sample(list(train_val_set - set(idx_train)), number_val)
    idx_test = list(map(int, list(test_set)))
    
    return idx_train, idx_val, idx_test

    
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def broken_graph(graph):
    for i in range(len(graph)):
        if len(graph[i]) < 2:
            continue
        j = graph[i][len(graph[i]) - 1]
        if len(graph[j]) < 2:
            continue
        else:
            graph[i].remove(j)
            graph[j].remove(i)
    return graph


def load_data(dataset_str, label_num):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph_T = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[:, 0] = 1
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    features = np.array(features.todense())
        
    adj_T = nx.adjacency_matrix(nx.from_dict_of_lists(graph_T))
    adj_T = preprocess_adj(adj_T)
    #adj_T = normalize(adj_T + sp.eye(adj_T.shape[0]))

    graph_S = broken_graph(graph_T)
    graph_S = broken_graph(graph_S)
    
    adj_S = nx.adjacency_matrix(nx.from_dict_of_lists(graph_S))
    #adj_S = normalize(adj_S + sp.eye(adj_S.shape[0]))
    adj_S = preprocess_adj(adj_S)
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #idx_train, idx_val, idx_test = select_sample(labels, labels.shape[0], label_num, 500, 1000, labels.shape[1]) # For cora
    idx_train, idx_val, idx_test = select_sample2(labels, labels.shape[0], set(test_idx_range), label_num, 500, labels.shape[1])
    
    adj_T = sparse_mx_to_torch_sparse_tensor(adj_T)
    adj_S = sparse_mx_to_torch_sparse_tensor(adj_S)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
   
    return adj_T, adj_S, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
