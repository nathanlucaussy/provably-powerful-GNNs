from .model_wrapper import ModelWrapper
from utils import one_hot_to_ints
import networkx as nx
from models import GIN

class GINWrapper(ModelWrapper):
    
    def __init__(self, dataset, config):
        super(GINWrapper, self).__init__(dataset, config)
        
        
    def transform_data(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        node_features = data.x
        label = int(data.y[0])
        node_tags = one_hot_to_ints(data.x)
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        nx_graph.add_edges_from(edge_index.transpose(0,1))
        
        neighbors = [[] for i in range(num_nodes)]
        for i, j in nx_graph.edges():
            neighbors[i].append(j)
            neighbors[j].append(i)
        degree_max = 0
        for i in range(num_nodes):
            degree_max = max(len(nx_graph.neighbors[i]), degree_max) 
        
        g = S2VGraph(nx_graph, label, node_tags)
    
        g.neighbors = neighbors
        g.node_features = node_features
        g.edge_mat = edge_index
        g.max_neighbor = degree_max
    
    
        return g
    
    def run(self):
        return 0.5 # TODO: Obviously needs changing!