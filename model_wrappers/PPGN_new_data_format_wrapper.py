from .model_wrapper import ModelWrapper
from models.PPGN_variants import new_data_format
from dataclasses import dataclass

class PPGNNewDataFormatWrapper(ModelWrapper):
    
    LEARNING_RATES = {'COLLAB': 0.0001, 'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001, 'QM9': 0.0001}
    DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0, 'QM9': 0.5}
    EPOCHS = {'COLLAB': 150, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC': 400, 'QM9': 500}
    
    @dataclass
    class Config:
        lr = 0.0001
        decay = 0.5
        epochs = 100
        print_freq = 20
        batch_size = 64
        param_search = False
        verbose = False
        
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

        self.model = new_data_format.PPGN
     
    # transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models and *graph label*
    def transform_data(self, data):
        return new_data_format.transform(data)
    
    def run(self):
        # For now, we won't allow param search on qm9
        if self.qm9:
            accuracy = new_data_format.CV_regression(self.model, self.data, self.config)
        elif self.config.param_search:
            lr, decay, accuracy = new_data_format.param_search(self.model, self.data, self.config)
            print(f'\nPARAMETER SEARCH COMPLETE. ACHIEVED BEST ACCURACY OF {accuracy} with lr={lr}, decay={decay}')
        else:
            accuracy = new_data_format.CV_10(self.model, self.data, self.config)
        return accuracy
    