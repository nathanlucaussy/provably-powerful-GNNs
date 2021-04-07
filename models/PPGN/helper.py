import numpy as np
from random import shuffle

def get_batches(graphs, batch_size):
    print("converting to list and grouping")
    grouped = group_same_size(list(graphs))
    print("shuffling groups")
    shuffle_sublists(grouped)
    print("splitting groups into batches")
    batches = split_to_batches(grouped, batch_size)
    print("shuffling batches")
    shuffle(batches)
    print("restructuring")
    # This final zip puts X and y together
    return [list(zip(*batch)) for batch in batches]

def group_same_size(graphs):
    graphs.sort(key = lambda g : g[0].shape[1])
    r_graphs = []
    same_size = []
    size = graphs[0][0].shape[1]
    for i in range(len(graphs)):
        if graphs[i][0].shape[1] == size:
            same_size.append(graphs[i])
        else:
            r_graphs.append(same_size)

            same_size = [graphs[i]]
            size = graphs[i][0].shape[1]

    r_graphs.append(same_size)
    return r_graphs


# helper method to shuffle each subarray
def shuffle_sublists(graphs):
    for same_size_graphs in graphs:
        shuffle(same_size_graphs)

def split_to_batches(graphs, batch_size):
    r_graphs = []
    for same_size_graphs in graphs:
        batch_count = 0
        curr_batch = []
        for i, graph in enumerate(same_size_graphs):
            if batch_count == batch_size:
                r_graphs.append(curr_batch)
                curr_batch = [graph]
                batch_count = 1
            else:
                curr_batch.append(graph)
                batch_count += 1
        r_graphs.append(curr_batch)
        
    return r_graphs
    # r_graphs = []
    # r_labels = []
    # for k in range(len(graphs)):
    #     r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
    #     r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])

    # # Avoid bug for batch_size=1, where instead of creating numpy array of objects, we had numpy array of floats with
    # # different sizes - could not reshape
    # ret1, ret2 = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
    # ret1[:] = r_graphs
    # ret2[:] = r_labels
    # return ret1, ret2

def normalize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)