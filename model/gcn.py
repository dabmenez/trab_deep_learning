import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN3Conv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Camada extra para melhor performance
        self.dropout = torch.nn.Dropout(dropout)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch
        # Primeira camada convolucional
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Segunda camada convolucional
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Terceira camada (extra para melhor performance)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Pooling global
        x = global_mean_pool(x, batch)
        
        # Camadas lineares
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

# https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275/
class GCN2Conv(torch.nn.Module):
    def __init__(self, input_ch, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_ch, 16)
        self.conv2 = GCNConv(16, 32)
        self.lin = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        output = self.lin(x)

        return output 