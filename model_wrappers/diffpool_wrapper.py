from .model_wrapper import ModelWrapper
from utils import one_hot_to_ints, cross_val_generator
import networkx as nx
import os
import sys
import torch
from dataclasses import dataclass


import numpy as np
import torch.utils.data


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
        bn = False
        #Whether link prediction side objective is used
        linkpred = False
        bias = True
        #the default is 1000?
        num_epochs = 350
        batch_size = 32
        iters_per_epoch = 50
        
        lr = 0.01
        filename = ''
        
    config = Config()
    
    def __init__(self, dataset, config):
        super(DiffPoolWrapper, self).__init__(dataset, config)
        self.download_repo(self.repo_url, 'DiffPool')
        DiffPool_root_dir = os.path.join(self.model_dir, 'diffpool-master')
        DiffPool_model_dir = os.path.join(DiffPool_root_dir, 'models')
        sys.path.insert(0, DiffPool_root_dir)
        sys.path.append(DiffPool_model_dir)
        import train
        self.DiffPool_main = train
        
    def transform_data(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        label = data.y[0]
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
                G.nodes[u]['feat'] = node_features[u]
        for u in G.nodes():
                G.nodes[u]['label'] = node_labels[u]
        G.graph['feat_dim'] = node_features[0].shape[0]

        # relabeling
        mapping={}
        it=0
        for n in G.nodes():
            mapping[n]=it
            it+=1
            
        # indexed from 0
        H = nx.relabel_nodes(G, mapping)

        s = GraphSampler([H], normalize=False, max_num_nodes=1000, features='default')

        return s[0]
    
    def run(self):
        conf = self.config
        device = conf.device

        graphs = self.data
        num_classes = self.data.num_classes
        output_dim = num_classes
        input_dim = graphs[0]['feats'].shape[1]

        model = self.DiffPool_main.encoders.SoftPoolingGcnEncoder(
                    conf.max_nodes, 
                    input_dim, conf.hidden_dim, output_dim, num_classes, conf.num_gc_layers,
                    conf.hidden_dim, assign_ratio=conf.assign_ratio, num_pooling=conf.num_pool,
                    bn=conf.bn, dropout=0.0, linkpred=conf.linkpred, args=conf,
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
        
    
# this is their code
class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=False, assign_feat='default', max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        
        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        #if features == 'default':
        self.feat_dim = G_list[0].nodes[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = G.nodes[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>max_deg] = max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = G.nodes[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in G.nodes[0]:
                    node_feats = np.array([G.nodes[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.assign_feat_all.append(self.feature_all[-1])
            
        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj':torch.tensor([adj_padded]),
                'feats':torch.tensor([self.feature_all[idx].copy()]),
                'label':torch.tensor([self.label_all[idx]]),
                'num_nodes': torch.tensor([num_nodes]),
                'assign_feats':torch.tensor(self.assign_feat_all[idx].copy())}
