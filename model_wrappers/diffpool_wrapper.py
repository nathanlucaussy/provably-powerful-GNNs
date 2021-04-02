from .model_wrapper import ModelWrapper
from utils import one_hot_to_ints, cross_val_generator
import networkx as nx
import os
import sys
import torch
from dataclasses import dataclass

class DiffPoolWrapper(ModelWrapper):
    
    repo_url = 'https://github.com/RexYing/diffpool/archive/refs/heads/master.zip'

    @dataclass
    class Config:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #max_nodes depends on the dataset?
        max_nodes = 1000
        hidden_dim = 20
        num_gc_layers = 3
        #ratio of number of nodes in consecutive layers - they had it 0.1 for both datasets, the default 0.25
        assign_ratio = 0.1
        num_pool = 1
        #Whether batch normalization is used
        bn = True
        #Whether link prediction side objective is used
        linkpred = False

        batch_size = 32
        iters_per_epoch = 50
        epochs = 350
        lr = 0.01
        seed = 0
        fold_idx = 0
        num_layers = 5
        num_mlp_layers = 2
        final_dropout = 0.5
        graph_pooling_type = 'sum'
        learn_eps = True
        degree_as_tag = True
        filename = ''
        
    config = Config()
    
    def __init__(self, dataset, config):
        super(DiffPoolWrapper, self).__init__(dataset, config)
        self.download_repo(self.repo_url, 'DiffPool')
        DiffPool_root_dir = os.path.join(self.model_dir, 'diffpool-master')
        DiffPool_model_dir = os.path.join(GIN_root_dir, 'models')
        sys.path.insert(0, DiffPool_root_dir)
        sys.path.append(DiffPool_model_dir)
        import train
        self.DiffPool_main = train
        
    def transform_data(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        label = int(data.y[0])
        node_labels = one_hot_to_ints(data.x)
        node_features = data.x

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for u, v in edge_index.transpose(0,1):
            G.add_edge(u.item(), v.item())
        # if max_nodes is not None and G.number_of_nodes() > max_nodes:
        #    continue
        G.graph['label'] = label 
        for u in G.nodes():
                G.node[u]['feat'] = node_features[u]
        for u in G.nodes():
                G.node[u]['label'] = node_labels[u]
        G.graph['feat_dim'] = node_features[0].shape[0]

        # relabeling
        mapping={}
        it=0
        for n in G.nodes():
            mapping[n]=it
            it+=1
            
        # indexed from 0
        H = nx.relabel_nodes(G, mapping)

        return H
    
    def run(self):
        conf = self.config
        device = conf.device

        graphs = self.data
        num_classes = self.data.num_classes
        output_dim = num_classes
        input_dim = graphs[0].graph['feat_dim']

        model = self.DiffPool_main.encoders.SoftPoolingGcnEncoder(
                    conf.max_nodes, 
                    input_dim, conf.hidden_dim, output_dim, num_classes, conf.num_gc_layers,
                    conf.hidden_dim, assign_ratio=conf.assign_ratio, num_pooling=conf.num_pool,
                    bn=conf.bn, dropout=0.0, linkpred=conf.linkpred, args=args,
                    assign_input_dim=-1).to(device)
        

    

        all_vals = []
        for train_graphs, test_graphs in cross_val_generator(graphs, 10):
            _, val_accs = self.DiffPool_main.train(train_graphs, model, conf, val_dataset=None, test_dataset=test_graphs, writer=None)
            all_vals.append(np.array(val_accs))


        
        all_vals = np.vstack(all_vals)
        all_vals = np.mean(all_vals, axis=0)
        print(all_vals)
        print(np.max(all_vals))
        print(np.argmax(all_vals))
        return all_vals
        
    
