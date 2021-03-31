import torch
import torch.nn as nn

class PPGN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Hard-coded config values
        self.new_suffix = True 
        block_features = [400, 400, 400]
        
        node_labels = config['node_labels']
        num_classes = config['num_classes']
        num_orig_features = node_labels + 1


        prev_layer_features = num_orig_features
        self.blocks = nn.ModuleList()
        for next_layer_features in block_features:
            self.blocks.append(RegBlock(prev_layer_features, next_layer_features))
            prev_layer_features = next_layer_features

        self.fc_layers = nn.ModuleList()
        if self.new_suffix:
            for out_features in block_features:
                self.fc_layers.append(FullyConnected(2*out_features, num_classes, activation=None))
        else:
            self.fc_layers.append(FullyConnected(2*block_features[-1], 512))
            self.fc_layers.append(FullyConnected(512, 256))
            self.fc_layers.append(FullyConnected(256, num_classes, activation=None))

    def forward(self, x):
        scores = torch.tensor(0, device=x.device, dtype=x.dtype)

        for i, block in enumerate(self.blocks):
            x = block(x)

            if self.new_suffix:
                scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores

        if not self.new_suffix:
            x = diag_offdiag_maxpool(x)
            for layer in self.fc_layers:
                x = layer(x)
            scores = x

        return scores

def diag_offdiag_maxpool(x):
    N = x.shape[-1]

    max_diag = torch.max(torch.diagonal(x, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * x)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=x.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(x - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S

class RegBlock(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()

        self.mlp1 = MlpBlock(in_features, out_features)
        self.mlp2 = MlpBlock(in_features, out_features)
        self.skip = Skip(in_features + out_features, out_features)

    def forward(self, input):
        mlp1 = self.mlp1(input)
        mlp2 = self.mlp2(input)

        return self.skip(input, torch.matmul(mlp1, mlp2))

class MlpBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        # Hardcoded
        depth = 2

        self.activation = nn.functional.relu
        self.convs = nn.ModuleList()

        for i in range(depth):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, x):
        out = x
        for conv in self.convs:
            out = self.activation(conv(out))

        return out

class Skip(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out

class FullyConnected(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation

    def forward(self, x):
        out = self.fc(x)
        if self.activation is not None:
            out = self.activation(out)

        return out

def _init_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)