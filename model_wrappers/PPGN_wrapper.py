from .model_wrapper import ModelWrapper
from models import PPGN
from utils import one_hot_to_ints, to_adj_mat_with_features

class PPGNWrapper(ModelWrapper):
    
    config = {
        'lr' : 0.0005,
        'epochs': 100,
        'print_freq': 20
    }
    
    def __init__(self, dataset, config):
        super(PPGNWrapper, self).__init__(dataset, config)
        
        self.config['node_labels'] = self.data.num_node_labels
        self.config['num_classes'] = self.data.num_classes
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
        accuracy = PPGN.model_utils.CV_10(self.model, self.data, self.config)
        return accuracy
    