import numpy as np
import torch
import scipy.spatial.distance as dist

#convert a tensor of one-hot vectors to a tensor of ints
def one_hot_to_ints(tensor):
    out_array = np.zeros(shape=(tensor.shape[0], 1))
    for row_index, row in enumerate(tensor):
        for col_index, entry in enumerate(row):
            if entry == 1.0:
                out_array[row_index] = col_index
    return(torch.from_numpy(out_array))

# takes a torch_geometric style adjacency list, node and edge features, and node positions (if they exist)
# outputs matrix for input to PPGN-style models
def to_adj_mat_with_features(sparse_mat, num_nodes, has_node_features, has_edge_features, has_node_positions,
                             node_features=None, edge_features=None, node_pos=None, 
                             num_node_features=0, num_edge_features=0, norm=False):

    num_total_features = num_node_features + num_edge_features + 1

    if has_node_positions:
        dist_matrix = dist.squareform(dist.pdist(node_pos))
        num_total_features += 1

    adj_and_features_array = np.zeros(shape=(num_nodes, num_nodes, num_total_features), dtype=np.float32)
            
    for i in range(len(sparse_mat[0])):
        adj_and_features_array[sparse_mat[0][i]][sparse_mat[1][i]][0] = 1
    
    if has_node_features:
        for i in range(num_nodes):
            adj_and_features_array[i][i][1:num_node_features + 1] = node_features[i, :]
    
    # add edge features (if they exist)
    if has_edge_features:
        for index in range(len(sparse_mat[0])):
            i, j = sparse_mat[:, index]
            adj_and_features_array[i][j][num_node_features + 1: num_total_features - 1] = edge_features[index]

    # add distance matrix on index [-1]
    if has_node_positions:
        adj_and_features_array[:,:,-1] = dist_matrix
        
    if norm:
        adj_and_features_array = normalize_graph(adj_and_features_array)

    transposed = np.transpose(adj_and_features_array, [2,0,1])
    return(torch.from_numpy(transposed))


# Given a list of data, return a cross-validation generator object 
def cross_val_generator(data, num_parts):
    parts = partition(data, num_parts)
    return (train_test_parts(parts, test_idx) for test_idx in range(num_parts))

def train_test_parts(parts, test_idx):
    test_part = parts[test_idx]
    train_parts = []
    for i, part in enumerate(parts):
        if i != test_idx:
            train_parts += part
    return train_parts, test_part
    
def partition(dataset, num_parts):
    N = len(dataset)
    part_size = N // num_parts
    mod = N % num_parts
    partitions = []
    count = 0
    part_start = 0
    while count < mod:
        part_end = part_start + part_size + 1
        partitions.append(dataset[part_start:part_end])
        part_start = part_end
        count += 1
    while count < num_parts:
        part_end = part_start + part_size
        partitions.append(dataset[part_start:part_end])
        part_start = part_end
        count += 1
    
    return partitions

def mean_and_std(train_set):
    all_labels = torch.cat([train_set.get(i).y for i in range(len(train_set))], dim=0)
    train_labels_mean = all_labels.mean(0)
    train_labels_std = all_labels.std(0)

    return train_labels_mean, train_labels_std

def normalize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1), dtype=np.float32), split[1], axis=2)
    return np.add(spred_adj, labels)