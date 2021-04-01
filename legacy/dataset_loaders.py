import csv
import torch
import networkx as nx
import numpy as np
import torch_geometric as tg
from util_GIN import transform_tg_to_gin_data
from DGCNN_utils import transform_tg_to_dgcnn_data

#LIST OF DATASET NAMES
dataset_names = ['MUTAG', 'PROTEINS']

#MAIN DATA LOADING FUNCTION
def load_dataset(ds_name):
    return(tg.datasets.TUDataset('./tg_datasets/', ds_name, transform=transform_tg_to_ppgn_data, use_node_attr=True))

#data loading function for GIN model
def load_dataset_GIN(ds_name):
    return(tg.datasets.TUDataset('./tg_datasets/', ds_name, transform=transform_tg_to_gin_data, use_node_attr=True))

#data loading function for DGCNN models
def load_dataset_DGCNN(ds_name):
    return(tg.datasets.TUDataset('./tg_datasets/', ds_name, transform=transform_tg_to_dgcnn_data, use_node_attr=True))

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
    transposed = np.transpose(adj_and_features_array, [2,0,1])
    return(torch.from_numpy(transposed))

# transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models and *graph label*
def transform_tg_to_ppgn_data(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    node_labels = one_hot_to_ints(data.x)
    num_node_labels = data.x[0].shape[0]
    #graph_label = torch.tensor([0, 1]) if int(data.y) == 1 else torch.tensor([1, 0])
    graph_label = int(data.y)
    return((transform_to_adjacency_matrix_with_features(edge_index, node_labels, num_nodes, num_node_labels), graph_label))


#DEPRECATED - method to load datasets in practical-style format
def old_load_dataset(name, has_node_features):
    #load dataset portions from csv files - **modelled on the MUTAG dataset format**
    with open('./datasets/'+name+'/edges', newline='') as f:
        reader = csv.reader(f)
        edges = []
        for record in reader:
            edges.append((float(record[0]), float(record[1])))
    with open('./datasets/'+name+'/graph_idx', newline='') as f:
        reader = csv.reader(f)
        graph_indices = []
        for record in reader:
            graph_indices.append(int(record[0]))

    with open('./datasets/'+name+'/graph_labels', newline='') as f:
        reader = csv.reader(f)
        graph_labels = []
        for record in reader:
            graph_labels.append(int(record[0]))

    if has_node_features:
        with open('./datasets/'+name+'/node_labels', newline='') as f:
            reader = csv.reader(f)
            node_labels = []
            for record in reader:
                node_labels.append(int(record[0]))

    #parse graphs from CSVs and create torch_geometric dataset
    node_index = 0
    dataset = []
    for graph_index in range(graph_indices[-1]):
        #collate all edges of graph, and node labels
        cur_graph_edges = []
        cur_node_labels = []
        while graph_indices[node_index] == graph_index:
            cur_graph_edges.append(edges[node_index])
            cur_node_labels.append(node_labels[node_index])
            node_index += 1

        #create networkx graph
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(cur_graph_edges)

        #convert to torch_geometric graph
        tg_graph = tg.utils.from_networkx(nx_graph)

        if has_node_features:
            tg_graph.x = torch.tensor(cur_node_labels)

        tg_graph.y = torch.tensor(graph_labels[graph_index])
        dataset.append(tg_graph)
    return dataset
