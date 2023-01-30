import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
# from conv_base import GNN_node_Virtualnode
from gnn_syn import GNNSynEncoder
from gnn_mol import GINMolHeadEncoder, vGINMolHeadEncoder
import pdb


class GINNet(torch.nn.Module):

    def __init__(self, num_class, 
                       dataset, 
                       in_dim=None, 
                       emb_dim=300,
                       num_layer=3,
                       dropout_rate=0.5, 
                       args=None):
        
        super(GINNet, self).__init__()
        # # pdb.set_trace()
        self.args = args
        self.dataset = dataset
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.dropout_rate = dropout_rate
        if dataset in ["motif", "cmnist"]:
            self.gnn_node = GNNSynEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        else:
            if self.args.virtual:
                self.gnn_node = vGINMolHeadEncoder(num_layer, emb_dim)
            else:    
                self.gnn_node = GINMolHeadEncoder(num_layer, emb_dim)

        self.pool = global_mean_pool
        self.classifier = torch.nn.Linear(emb_dim, num_class)
        self.predictor = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim), 
                nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(), 
                nn.Dropout(),
                nn.Linear(2 * emb_dim, num_class))


    def forward(self, batched_data, return_feature=False):
        # # pdb.set_trace()
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        if self.dataset in ["motif", "cmnist"]:
            h_node = self.gnn_node(x, edge_index)
        else:
            h_node = self.gnn_node(x, edge_index, edge_attr, batch)

        h_graph = self.pool(h_node, batch)
        if return_feature:
            return h_graph
        if self.args.use_linear:
            return self.classifier(h_graph)
        else:
            return self.predictor(h_graph)