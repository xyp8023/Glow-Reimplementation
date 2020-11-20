import numpy as np
import torch
import torch.nn as nn

from .utils import LinearZeros ,Graph, st_gcn, GraphConvolution

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, 
                 num_layers=2, dropout=0.0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = LinearZeros(hidden_dim, output_dim)
        self.do_init = True
    
    def init_hidden(self):
        self.do_init = True
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.do_init:
            lstm_out, self.hidden = self.lstm(x)
            self.do_init = False
        else:
            lstm_out, self.hidden = self.lstm(x, self.hidden)
        
        y = self.linear(lstm_out).permute(0, 2, 1)
        return y        


# class STGCN(nn.Module):
#     def __init__(self, input_dim, output_dim=1,
#                  layout='locomotion'):
#         super().__init__()
#         graph = Graph(layout=layout)
#         A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
#         self.register_buffer('A', A)

#         spatial_kernel_size = A.size(0)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, 
                 layout='locomotion', graph_scale=1.0,
                 edge_importance_weighting=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        graph = Graph(layout=layout, scale=graph_scale)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        kernel_size = A.size(0)
        
        self.gcn_networks = nn.ModuleList((
            GraphConvolution(input_dim, hidden_dim, kernel_size),
            GraphConvolution(hidden_dim, hidden_dim, kernel_size)
        ))
        
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
                for i in self.gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_networks)
            
        self.linear = LinearZeros(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.unsqueeze(2) # N, C, 1, V
        for gcn, importance in zip(self.gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
        x = x.squeeze(2).permute(0, 2, 1)
        y = self.linear(x).permute(0, 2, 1)
        return y