import networkx as nx
import torch_geometric as tg
import torch
from utils import one_hot_to_ints, to_adj_mat_with_features
from torch_geometric.data import Dataset
import random 

class customDataset(Dataset):
	def __init__(self, transform=None, pre_transform=None):
		super(customDataset, self).__init__(None, transform, pre_transform)
		self.n1 = 6
		self.n2 = 20
		self.data = cycleDataset(self.n1, self.n2)
		self.num_classes = 2
		self.num_node_labels = 25


	def get(self, id):
		return self.data[id]

	#def get(self):
		#self.data

	#def __getitem__(self, i):
		#self.data[i]

	#def __iter__(self):
		#return (x for x in enumerate(self.data))

	def len(self):
		return len(self.data)

	#def __len__(self):
		#return len(self.data)

def cycleDataset(n1, n2): 
	s = set()
	for n in range(n1, n2):
	    for i in range(3, n-2):
	        G = graph_cycle(n)
	        H = two_cycles(i, n-i)
	        s.add((G, H))
	data = set()
	for (G, H) in s:
	    G1 = tg.utils.from_networkx(G)
	    H1 = tg.utils.from_networkx(H)
	    G1.x = torch.zeros([G.number_of_nodes(), 25], dtype=torch.float32)
	    G1.y = torch.tensor([1])
	    H1.x = torch.zeros([H.number_of_nodes(), 25], dtype=torch.float32)
	    H1.y = torch.tensor([0])

	    #data.add((transform_data(G1),transform_data(H1)))
	    data.add((transform_data(G1),transform_data(H1)) )
	    #data.add(transform_data(H1))
	
	#print(type(dataset))

	dataset = list(data)
	#print(dataset[0])
	random.shuffle(dataset)
	return dataset

 # transform a torch_geometric.data.Data object to the matrix needed for PPGN-style models and *graph label*
def transform_data(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    node_labels = data.x
    num_node_labels = data.x[0].shape[0]
    graph_label = int(data.y)
    #print(data.y)

    return((to_adj_mat_with_features(edge_index, num_nodes, True, False, False, 
                                     node_features=node_labels, num_node_features=num_node_labels),
            graph_label))

def two_cycles(n_1, n_2):
    G = nx.Graph()
    G.add_nodes_from(range(1, n_1+1))
    G.add_edges_from(zip(range(1,n_1), range(2,n_1+1)))
    G.add_edge(1,n_1)
    F = nx.Graph()
    F.add_nodes_from(range(1 + n_1, n_1 + 1 + n_2))
    F.add_edges_from(zip(range(1 + n_1, n_1 + n_2), range(2 + n_1 , n_1 + n_2 + 1)))
    F.add_edge(1 + n_1, n_1 + n_2)
    return nx.union(G, F)

def graph_cycle(n):
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from(zip(range(1, n), range(2, n + 1)))
    G.add_edge(1, n)
    return G

 