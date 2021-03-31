from torch_geometric.datasets import TUDataset
import os
import sys

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
        