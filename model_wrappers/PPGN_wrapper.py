from .model_wrapper import ModelWrapper
from models import PPGN
from utils import one_hot_to_ints, to_adj_mat_with_features
from dataclasses import dataclass

class PPGNWrapper(ModelWrapper):
    
    LEARNING_RATES = {'COLLAB': 0.0001, 'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001}
    DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0}
    EPOCHS = {'COLLAB': 150, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC': 400}
    
    @dataclass
    class Config:
        lr = 0.0001
        decay = 0.5
        epochs = 100
        print_freq = 20
        param_search = False
        verbose = False
        
    config = Config()
    
    def __init__(self, dataset, config):
        if dataset in self.LEARNING_RATES:
            self.config.lr = self.LEARNING_RATES[dataset]
            self.config.decay = self.DECAY_RATES[dataset]
            self.config.epochs = self.EPOCHS[dataset]
        super(PPGNWrapper, self).__init__(dataset, config)
        self.config.node_labels = self.data.num_node_labels
        self.config.num_classes = self.data.num_classes
        self.model = PPGN.ppgn.PPGN(self.config)                
     
    # transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models and *graph label*
    def transform_data(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        node_labels = one_hot_to_ints(data.x)
        num_node_labels = data.x[0].shape[0]
        graph_label = int(data.y)
        return((to_adj_mat_with_features(edge_index,
                                               node_labels,
                                               num_nodes,
                                               num_node_labels),
                graph_label))
    
    def run(self):
        if self.config.param_search:
            lr, decay, acc = PPGN.model_utils.param_search(self.model, self.data, self.config)
            print('\nPARAMETER SEARCH COMPLETE. ACHIEVED BEST ACCURACY OF {acc} with lr={lr}, decay={decay}')
            return acc
        else:
            acc = PPGN.model_utils.CV_10(self.model, self.data, self.config)
        return acc
    