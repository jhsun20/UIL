import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from MolEncoders import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
import math
import pdb

nn_act = torch.nn.ReLU() #ReLU()
F_act = F.relu

        
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight=None):

        if edge_weight is not None:
            mess = F_act((x_j + edge_attr) * edge_weight)
        else:
            mess = F_act(x_j + edge_attr)
        return mess

        # return F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        
        super(GINEConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), 
                                       torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight=None):
        if edge_weight is not None:
            mess = F.relu((x_j + edge_attr) * edge_weight)
        else:
            mess = F.relu(x_j + edge_attr)
        return mess

    def update(self, aggr_out):
        return aggr_out

class GINMolHeadEncoder(torch.nn.Module):
 
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=True):
        
        super(GINMolHeadEncoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training = self.training)
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation



class vGINMolHeadEncoder(torch.nn.Module):
    
    def __init__(self, num_layer, emb_dim):
        super(vGINMolHeadEncoder, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.atom_encoder = AtomEncoder(emb_dim)
        self.conv1 = GINEConv(emb_dim)
        self.convs = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer - 1)])
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [nn.ReLU() for _ in range(num_layer - 1)]
        )
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(emb_dim)
            for _ in range(num_layer - 1)
        ])
        self.dropout1 = nn.Dropout()
        self.dropouts = nn.ModuleList([
            nn.Dropout() for _ in range(num_layer - 1)
        ])

        self.virtual_node_embedding = nn.Embedding(1, emb_dim)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(emb_dim, 2 * emb_dim),
                 nn.BatchNorm1d(2 * emb_dim), nn.ReLU()] +
                [nn.Linear(2 * emb_dim, emb_dim),
                 nn.BatchNorm1d(emb_dim), nn.ReLU(),
                 nn.Dropout()]
        ))
        self.virtual_pool = global_add_pool

    def forward(self, x, edge_index, edge_attr, batch):

        virtual_node_feat = self.virtual_node_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)
        # out_readout = self.readout(post_conv, batch)
        return post_conv

