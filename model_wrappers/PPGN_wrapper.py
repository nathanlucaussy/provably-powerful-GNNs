from .model_wrapper import ModelWrapper
from models import PPGN
from utils import one_hot_to_ints, to_adj_mat_with_features
from dataclasses import dataclass

class PPGNWrapper(ModelWrapper):
    
    LEARNING_RATES = {'COLLAB': 0.0001, 'IMDB-BINARY': 0.00005, 'IMDB-MULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC_FM': 0.0001, 'QM9': 0.0001}
    DECAY_RATES = {'COLLAB': 0.5, 'IMDB-BINARY': 0.5, 'IMDB-MULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC_FM': 1.0, 'QM9': 0.5}
    EPOCHS = {'COLLAB': 150, 'IMDB-BINARY': 100, 'IMDB-MULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC_FM': 400, 'QM9': 500}
    
    @dataclass
    class Config:
        lr = 0.0001
        decay = 0.5
        epochs = 100
        print_freq = 20
        batch_size = 1
        param_search = False
        verbose = False
        
    config = Config()
    
    def __init__(self, dataset, config):
        if dataset in self.LEARNING_RATES:
            self.config.lr = self.LEARNING_RATES[dataset]
            self.config.decay = self.DECAY_RATES[dataset]
            self.config.epochs = self.EPOCHS[dataset]
        super(PPGNWrapper, self).__init__(dataset, config)
        self.config.qm9 = self.qm9

        if self.config.qm9:
            X, y = self.data[0]
            self.config.input_size = X.shape[0]
            self.config.output_size = y.shape[1]
        else:
            self.config.input_size = self.data.num_node_labels + 1
            self.config.output_size = self.data.num_classes

        #self.config.node_labels = self.data.num_node_labels
        #self.config.num_classes = self.data.num_classes
        self.model = PPGN.ppgn.PPGN
     
    # transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models and *graph label*
    def transform_data(self, data):
        if self.qm9:
            return self.transform_data_qm9(data)
        else:
            return self.transform_data_classification(data)      
     
    # transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models 
    # when doing classification, and *graph label* 
    def transform_data_classification(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        num_node_labels = data.num_node_features
        has_node_labels = num_node_labels > 0
        node_labels = data.x

        graph_label = int(data.y)

        return((to_adj_mat_with_features(edge_index, num_nodes, has_node_labels, False, False, 
                                         node_features=node_labels, num_node_features=num_node_labels),
                graph_label))


    # transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models 
    # when doing regression on QM9, and the graph labels  
    def transform_data_qm9(self, data):
        edge_index = data.edge_index
        node_features = data.x
        num_nodes = node_features.shape[0]
        num_node_features = node_features.shape[1]
        edge_features = data.edge_attr
        num_edge_features = edge_features.shape[1]
        node_pos = data.pos

        return((to_adj_mat_with_features(edge_index, num_nodes, True, True, True,
                                         node_features=node_features, edge_features=edge_features, node_pos=node_pos,
                                         num_node_features=num_node_features, num_edge_features=num_edge_features),
                data.y))


    
    def run(self):       
        # For now, we won't allow param search on qm9
        if self.qm9:
            accuracy = PPGN.model_utils.CV_regression(self.model, self.data, self.config)
        elif self.config.param_search:
            lr, decay, accuracy = PPGN.model_utils.param_search(self.model, self.data, self.config)
            print(f'\nPARAMETER SEARCH COMPLETE. ACHIEVED BEST ACCURACY OF {accuracy} with lr={lr}, decay={decay}')
        else:
            accuracy = PPGN.model_utils.CV_10(self.model, self.data, self.config)
        return accuracy
    