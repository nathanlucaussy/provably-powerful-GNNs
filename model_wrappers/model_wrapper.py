from torch_geometric.datasets import TUDataset
import os
import sys
from urllib.request import urlopen
from zipfile import ZipFile
import importlib.util

class ModelWrapper:
    
    use_node_attr = True
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, '../tg_datasets')
    models_path = os.path.join(curr_dir, '../models/other')
    config = {} # Default config to be properly defined by child class
    model_dir = None
    
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
        