import csv
import torch
import networkx as nx
import torch_geometric as tg

#LIST OF DATASET NAMES
dataset_names = ['MUTAG', 'PROTEINS']

def load_dataset(name, has_node_features):
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
