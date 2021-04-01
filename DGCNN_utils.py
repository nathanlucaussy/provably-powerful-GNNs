from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import networkx as nx
import pdb
import argparse
import torch


class GNNGraph(object):
  def __init__(self, g, label, node_tags=None, node_features=None):
    '''
      g: a networkx graph
      label: an integer graph label
      node_tags: a list of integer node tags
      node_features: a numpy array of continuous node features
    '''
    self.num_nodes = len(node_tags)
    self.node_tags = node_tags
    self.label = label
    self.node_features = node_features  # numpy array (node_num * feature_dim)
    self.degs = list(dict(g.degree).values())

    if len(g.edges()) != 0:
      x, y = zip(*g.edges())
      self.num_edges = len(x)        
      self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
      self.edge_pairs[:, 0] = x
      self.edge_pairs[:, 1] = y
      self.edge_pairs = self.edge_pairs.flatten()
    else:
      self.num_edges = 0
      self.edge_pairs = np.array([])
        
    # see if there are edge features
    self.edge_features = None
    if nx.get_edge_attributes(g, 'features'):  
      # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
      edge_features = nx.get_edge_attributes(g, 'features')
      assert(type(edge_features.values()[0]) == np.ndarray) 
      # need to rearrange edge_features using the e2n edge order
      edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
      keys = sorted(edge_features)
      self.edge_features = []
      for edge in keys:
        self.edge_features.append(edge_features[edge])
        self.edge_features.append(edge_features[edge])  # add reversed edges
      self.edge_features = np.concatenate(self.edge_features, 0)

#convert a tensor of one-hot vectors to a tensor of ints
def one_hot_to_ints(tensor):
    out_array = np.zeros(shape=(tensor.shape[0], 1))
    for row_index, row in enumerate(tensor):
        for col_index, entry in enumerate(row):
            if entry == 1.0:
                out_array[row_index] = col_index
    return(torch.from_numpy(out_array))

def transform_tg_to_dgcnn_data(data):
  edge_index = data.edge_index
  num_nodes = data.num_nodes
  node_features = data.x
  label = int(data.y[0])
  node_tags = one_hot_to_ints(data.x)

  nx_graph = nx.Graph()
  nx_graph.add_nodes_from(range(num_nodes))
  nx_graph.add_edges_from(edge_index.transpose(0,1))
    
  g = GNNGraph(nx_graph, label, node_tags, node_features)
  
  return g 
