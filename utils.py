import numpy as np
import torch

#convert a tensor of one-hot vectors to a tensor of ints
def one_hot_to_ints(tensor):
    out_array = np.zeros(shape=(tensor.shape[0], 1))
    for row_index, row in enumerate(tensor):
        for col_index, entry in enumerate(row):
            if entry == 1.0:
                out_array[row_index] = col_index
    return(torch.from_numpy(out_array))

# takes a torch_geometric style adjacency list and node features.
# outputs matrix for input to PPGN-style models
def to_adj_mat_with_features(self, sparse_mat, node_labels, num_nodes, num_node_labels):
    adj_mat = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
    for index in range(len(sparse_mat[0])):
        adj_mat[sparse_mat[0][index]][sparse_mat[1][index]] = 1

    adj_and_features_array = np.zeros(shape=(num_nodes, num_nodes, num_node_labels + 1), dtype=np.float32)
    for row_index in range(num_nodes):

        #encode node label (feature) for node 'row_index' as a one-hot encoding at the self-loop
        #on indices [1 ... num_node_labels + 1]


        adj_and_features_array[row_index][row_index][int(node_labels[row_index][0])+1] = 1

        #encode adjacency matrix on index [0] of the innermost vectors / feature vector, indexed by edges
        for col_index in range(num_nodes):
            adj_and_features_array[row_index][col_index][0] = 1
    transposed = np.transpose(adj_and_features_array, [2,0,1])
    return(torch.from_numpy(transposed))