from torch_geometric.datasets import TUDataset, QM9
import os
import sys
from urllib.request import urlopen
from zipfile import ZipFile
from customDataset import customDataset

class ModelWrapper:
    
    use_node_attr = True
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../tg_datasets')
    models_path = os.path.join(curr_dir, '../models/other')
    model_dir = None
    
    def __init__(self, dataset, config):
        self.qm9 = False
        if dataset == "QM9":
            self.qm9 = True
            self.data_dir = os.path.join(self.curr_dir, '../tg_datasets/QM9')
            self.data = QM9(self.data_dir,
                            transform=self.transform_data)
        elif dataset == "CUSTOM":
            self.data = customDataset()
        else:
            self.data = TUDataset(self.data_dir,
                                  dataset,
                                  transform=self.transform_data,
                                  use_node_attr=self.use_node_attr)
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, type(getattr(self.config, key))(value))
            else:
                print(f'Config key \'{key}\' is not valid for PPGN')
                sys.exit()                
        
    # transform a torch_geometric.data.Data object to whatever format this model requires
    def transform_data(self, data):
        raise NotImplementedError()
        
    def run(self):
        raise NotImplementedError()
        
    def download_repo(self, url, model_name):
        self.model_dir = os.path.join(self.models_path, model_name)
        self.model_name = model_name
        if os.path.exists(self.model_dir):
            print(f'Directory for model {model_name} already exists, not downloading')
        else:
            print(f'Downloading model: {model_name}')
            zip_resp = urlopen(url)
            with open('temp.zip', 'wb') as temp_zip:
                temp_zip.write(zip_resp.read())
            zip_file = ZipFile('temp.zip')
            zip_file.extractall(path = self.model_dir)
            zip_file.close()
            os.remove('temp.zip')
        

    def import_model_dir(self):
        # Bit hacky but ah well
        exec(f'from models.other import {self.model_name}')
        