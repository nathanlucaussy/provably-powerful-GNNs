from .model_wrapper import ModelWrapper
from utils import one_hot_to_ints, cross_val_generator
import networkx as nx
import os
import sys
import torch
import random
import numpy as np
from dataclasses import dataclass

class DGCNNWrapper(ModelWrapper):
    
    repo_url = 'https://github.com/muhanzhang/pytorch_DGCNN/archive/refs/heads/master.zip'
    
    # Epochs/lr to use for each model. See run_DGCNN.sh in the repo.
    epochs_lr = {
        'MUTAG': (300, 0.0001)
    }

    @dataclass
    class Config:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lr = 0.0001
        epochs = 200
        seed = 1
        batch_size = 128

        
    config = Config()
    
    def __init__(self, dataset, config):
        super(DGCNNWrapper, self).__init__(dataset, config)
        self.download_repo(self.repo_url, 'DGCNN')
        DGCNN_root_dir = os.path.join(self.model_dir, 'pytorch_DGCNN-master')
        DGCNN_model_dir = os.path.join(DGCNN_root_dir, 'models')
        sys.path.insert(0, DGCNN_root_dir)
        sys.path.append(DGCNN_model_dir)
        import main
        self.DGCNN_main = main

        self.config.epochs, self.config.lr = self.epochs_lr[dataset]
        
    def transform_data(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        # Commented out the below for now as the model seems to get confused
        # when you provide both node features and node tags
        # node_features = np.array(data.x.float())
        node_features = None
        label = int(data.y[0])

        if data.x is not None: 
            node_tags = torch.squeeze(one_hot_to_ints(data.x).to(dtype=torch.long))
        else:
            ones = torch.ones(size=(num_nodes, 1))
            node_tags = torch.squeeze(ones.to(dtype=torch.long))

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        for u, v in edge_index.transpose(0,1):
            nx_graph.add_edge(u.item(), v.item())
        g = GNNGraph(nx_graph, label, node_tags, node_features)
        
        return g 
    
    def run(self):
        conf = self.config
        device = conf.device
    
        graphs = self.data
        num_classes = self.data.num_classes
        self.DGCNN_main.cmd_args.mode = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.DGCNN_main.cmd_args.num_class = num_classes
        self.DGCNN_main.cmd_args.feat_dim = self.data.num_node_labels
        self.DGCNN_main.cmd_args.edge_feat_dim = 0
        self.DGCNN_main.cmd_args.attr_dim = self.data.num_node_attributes
        self.DGCNN_main.cmd_args.latent_dim = [32, 32, 32, 1]
        self.DGCNN_main.cmd_args.batch_size = self.config.batch_size
        self.DGCNN_main.cmd_args.sortpooling_k = 19
        self.DGCNN_main.cmd_args.hidden = 128
        self.DGCNN_main.cmd_args.dropout = True
    
        model = self.DGCNN_main.Classifier().to(device)
    
        optimizer = self.DGCNN_main.optim.Adam(model.parameters(), lr=conf.lr)
        scheduler = self.DGCNN_main.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
        acc_sum = 0
        count = 1
        for train_graphs, test_graphs in cross_val_generator(graphs, 10):
            train_idxes = list(range(len(train_graphs)))
            print(f'Running with test set {count}/10')
            model.train()
            for epoch in range(1, conf.epochs + 1):
                scheduler.step()
                random.shuffle(train_idxes)

                self.DGCNN_main.loop_dataset(train_graphs, model, train_idxes, optimizer=optimizer)
                
            model.eval()
            acc_test = self.DGCNN_main.loop_dataset(test_graphs, model, list(range(len(test_graphs))))[1]
            acc_sum += acc_test
            count += 1
        avg_acc = acc_sum / 10
        return avg_acc
        
    
class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
          g: a networkx graph
          label: an integer graph label
          node_tags: a list of integer node tags
          node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
            
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(edge_features.values()[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)
