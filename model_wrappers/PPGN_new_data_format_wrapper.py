from .model_wrapper import ModelWrapper
#from models.PPGN_variants import new_data_format
from models.PPGN import PPGN, CV_regression, param_search, CV_10
from dataclasses import dataclass
import torch

class PPGNNewDataFormatWrapper(ModelWrapper):
    
    LEARNING_RATES = {'COLLAB': 0.0001, 'IMDB-BINARY': 0.00005, 'IMDB-MULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC_FM': 0.0001, 'QM9': 0.0001}
    DECAY_RATES = {'COLLAB': 0.5, 'IMDB-BINARY': 0.5, 'IMDB-MULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC_FM': 1.0, 'QM9': 0.8}
    EPOCHS = {'COLLAB': 150, 'IMDB-BINARY': 100, 'IMDB-MULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC_FM': 400, 'QM9': 500}
    
    @dataclass
    class Config:
        lr = 0.0001
        decay = 0.5
        epochs = 100
        print_freq = 20
        batch_size = 64
        param_search = False
        verbose = False
        block_feat = 400
        depth = 2
        new_suffix = True
        
    config = Config()
    
    def __init__(self, dataset, config):
        if dataset in self.LEARNING_RATES:
            self.config.lr = self.LEARNING_RATES[dataset]
            self.config.decay = self.DECAY_RATES[dataset]
            self.config.epochs = self.EPOCHS[dataset]
        super(PPGNNewDataFormatWrapper, self).__init__(dataset, config)
        self.config.qm9 = self.qm9

        if self.config.qm9:
            X, y = self.data[0]
            self.config.input_size = X.shape[0]
            self.config.output_size = y.shape[1]
        else:
            self.config.input_size = self.data.num_node_labels
            self.config.output_size = self.data.num_classes

        self.model = PPGN
     
    # transform a torch_geometric.data.Data object to the matrix needed for PPGNNewData-style models and *graph label*
    def transform_data(self, data):
        return self.transform(data)
    
    def run(self):
        # For now, we won't allow param search on qm9
        if self.qm9:
            accuracy = CV_regression(self.model, self.data, self.config)
        elif self.config.param_search:
            lr, decay, accuracy = param_search(self.model, self.data, self.config)
            print(f'\nPARAMETER SEARCH COMPLETE. ACHIEVED BEST ACCURACY OF {accuracy} with lr={lr}, decay={decay}')
        else:
            accuracy = CV_10(self.model, self.data, self.config)
        return accuracy
    
    def transform(self, data):
        num_nodes = data.num_nodes
        node_feats = data.x
        if node_feats is None:
            node_feats = torch.zeros((num_nodes, 1))
        num_node_feats = len(node_feats[0])
        edge_feats = data.edge_attr
        if edge_feats is None:
            edge_feats = torch.zeros((len(data.edge_index[0]), num_node_feats))
            edge_feats[:,0] = 1
            num_edge_feats = num_node_feats
        else:
            num_edge_feats = len(edge_feats[0])
        if num_edge_feats < num_node_feats:
            max_dim = num_node_feats
            diff = num_node_feats - num_edge_feats
            # Fill out edge_feats with extra dims
            edge_feats = torch.stack([torch.cat((e, torch.zeros(diff))) for e in edge_feats])
        elif num_node_feats < num_edge_feats:
            max_dim = num_edge_feats
            diff = num_edge_feats - num_node_feats
            node_feats = torch.stack([torch.cat((v, torch.zeros(diff))) for v in node_feats])
        else:
            max_dim = num_node_feats
            
        mat = torch.zeros(num_nodes, num_nodes, max_dim)
        for edge_feat, v1, v2 in zip(edge_feats, data.edge_index[0], data.edge_index[1]):
            mat[v1][v2] = edge_feat
            
        for v1, node_feat in enumerate(node_feats):
            mat[v1][v1] = node_feat
            
        return (mat.transpose(0, 1).transpose(0,2), int(data.y))
    