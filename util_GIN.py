import networkx as nx
import numpy as np
import random
import torch

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

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
def transform_to_adjacency_matrix_with_features(sparse_mat, node_labels, num_nodes, num_node_labels):
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
    return(torch.from_numpy(adj_and_features_array))


def transform_tg_to_gin_data(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    node_features = data.x
    label = int(data.y[0])
    node_tags = one_hot_to_ints(data.x)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edge_index.transpose(0,1))
    
    neighbors = [[] for i in range(num_nodes)]
    for i, j in nx_graph.edges():
        neighbors[i].append(j)
        neighbors[j].append(i)
    degree_list = []
    degree_max = 0
    for i in range(num_nodes):
        degree_max = max(len(nx_graph.neighbors[i]), degree_max) 
    
    g = S2VGraph(nx_graph, label, node_tags)

    g.neighbors = neighbors
    g.node_features = node_features
    g.edge_mat = edge_index
    g.max_neighbor = degree_max


    return g 




