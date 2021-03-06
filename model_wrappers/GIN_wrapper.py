from .model_wrapper import ModelWrapper
from utils import one_hot_to_ints, cross_val_generator
import networkx as nx
import os
import sys
import torch
from dataclasses import dataclass

class GINWrapper(ModelWrapper):
    
    repo_url = 'https://github.com/weihua916/powerful-gnns/archive/refs/heads/master.zip'
    
    @dataclass
    class Config:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 32
        iters_per_epoch = 50
        epochs = 350
        lr = 0.01
        seed = 0
        fold_idx = 0
        num_layers = 5
        num_mlp_layers = 2
        hidden_dim = 64
        final_dropout = 0.5
        graph_pooling_type = 'sum'
        neighbor_pooling_type = 'sum'
        learn_eps = True
        degree_as_tag = True
        filename = ''
        
    config = Config()
    
    def __init__(self, dataset, config):
        super(GINWrapper, self).__init__(dataset, config)
        self.download_repo(self.repo_url, 'GIN')
        GIN_root_dir = os.path.join(self.model_dir, 'powerful-gnns-master')
        GIN_model_dir = os.path.join(GIN_root_dir, 'models')
        sys.path.insert(0, GIN_root_dir)
        sys.path.append(GIN_model_dir)
        import main
        self.GIN_main = main
        
    def transform_data(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        if data.x is not None:
            node_features = data.x.float()
        else:
            node_features = torch.ones((num_nodes, 1)).float()

        node_tags = one_hot_to_ints(node_features)
        label = int(data.y[0])

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        for u, v in edge_index.transpose(0,1):
            nx_graph.add_edge(u.item(), v.item())
        
        neighbors = [[] for i in range(num_nodes)]
        for i, j in nx_graph.edges():
            neighbors[i].append(j)
            neighbors[j].append(i)
        degree_max = 0
        for i in range(num_nodes):
            degree_max = max(len(neighbors[i]), degree_max) 
        
        g = S2VGraph(nx_graph, label, node_tags)
    
        g.neighbors = neighbors
        g.node_features = node_features
        g.edge_mat = edge_index
        g.max_neighbor = degree_max
    
        return g
    
    def run(self):
        conf = self.config
        device = conf.device
    
        graphs = self.data
        num_classes = self.data.num_classes
    
        # original 10-fold validation code
        # train_graphs, test_graphs = self.GIN_main.separate_data(graphs, conf.seed, conf.fold_idx)
    
        acc_sum = 0
        for train_graphs, test_graphs in cross_val_generator(graphs, 10):
            model = self.GIN_main.GraphCNN(conf.num_layers, conf.num_mlp_layers, graphs[0].node_features.shape[1], conf.hidden_dim, num_classes, conf.final_dropout, conf.learn_eps, conf.graph_pooling_type, conf.neighbor_pooling_type, device).to(device)
            optimizer = self.GIN_main.optim.Adam(model.parameters(), lr=conf.lr)
            scheduler = self.GIN_main.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            for epoch in range(1, conf.epochs + 1):
                self.GIN_main.train(conf, model, device, list(train_graphs), optimizer, epoch)
                scheduler.step()
        
            acc_train, acc_test = self.GIN_main.test(conf, model, device, list(train_graphs), list(test_graphs), 0)
            acc_sum += acc_test

        avg_acc = acc_sum / 10
        return avg_acc
        
    
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
    