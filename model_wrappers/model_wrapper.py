from torch_geometric.datasets import TUDataset
import os

class ModelWrapper:
    
    use_node_attr = True
    data_dir = os.path.join(os.path.dirname(__file__), '../tg_datasets')
    config = {} # Default config to be properly defined by child class
    
    def __init__(self, dataset, config):
        self.data = TUDataset(self.data_dir,
                              dataset,
                              transform=self.transform_data,
                              use_node_attr=self.use_node_attr)
        for key in config:
            try:
                self.config[key] = type(self.config[key])(config[key])
            except KeyError:
                print(f'Config key \'{key}\' is not valid for PPGN')
                sys.exit()
        
    # transform a torch_geometric.data.Data object to whatever format this model requires
    def transform_data(self, data):
        raise NotImplementedError()
        
    def run(self):
        raise NotImplementedError()
        
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
    