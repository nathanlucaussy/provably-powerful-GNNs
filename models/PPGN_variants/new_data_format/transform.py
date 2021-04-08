import torch

def one_hot_to_ints(tensor):
    ints = torch.zeros(len(tensor))
    for row_index, row in enumerate(tensor):
        for col_index, entry in enumerate(row):
            if entry == 1.0:
                ints[row_index] = col_index
    return(ints)

def transform(data):
    edge_feats = data.edge_attr
    num_edge_feats = len(edge_feats[0])
    node_feats = data.x
    num_node_feats = len(node_feats[0])
    num_nodes = len(data.x)
    if num_edge_feats < num_node_feats:
        max_dim = num_node_feats
        diff = num_node_feats - num_edge_feats
        # Fill out edge_feats with extra dims
        edge_feats = torch.stack([torch.cat((e, torch.zeros(diff))) for e in edge_feats])
    elif num_node_feats < num_edge_feats:
        max_dim = num_edge_feats
        diff = num_edge_feats - num_node_feats
        node_feats = torch.stack([torch.cat((v, torch.zeros(diff))) for v in node_feats])
        
    mat = torch.zeros(num_nodes, num_nodes, max_dim)
    for edge_feat, v1, v2 in zip(edge_feats, data.edge_index[0], data.edge_index[1]):
        mat[v1][v2] = edge_feat
        
    for v1, node_feat in enumerate(node_feats):
        mat[v1][v1] = node_feat
        
    return (mat.transpose(0, 1).transpose(0,2), int(data.y))
    