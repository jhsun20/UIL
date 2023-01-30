import pdb
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_mol import (GINMolHeadEncoder, GNNMolTailEncoder, GraphMolMasker,
                      vGINMolHeadEncoder)
from conv_syn import GINConv, GNNSynEncoder, GraphSynMasker
from graphon import estimate_graphon
from torch_geometric.nn import MessagePassing, global_mean_pool
import time
import numpy as np


class CausalGraphon(torch.nn.Module):

    def __init__(self, args,
                       num_class, 
                       in_dim,
                       emb_dim=300,
                       fro_layer=2,
                       bac_layer=2,
                       cau_layer=2,
                       dropout_rate=0.5,
                       cau_gamma=0.4,
                       env_gamma=1.0,
                       use_linear=False,
                       graphon=True,
                       N=15):

        super(CausalGraphon, self).__init__()
        self.args = args
        self.cau_gamma = cau_gamma
        self.env_gamma = env_gamma

        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graphon = graphon
        self.N = N

        self.graph_front = GNNSynEncoder(fro_layer, in_dim,  emb_dim, dropout_rate)
        self.graph_backs = GNNSynEncoder(bac_layer, emb_dim, emb_dim, dropout_rate)
        self.causaler = GraphSynMasker(cau_layer, in_dim, emb_dim, dropout_rate)

        self.pool = global_mean_pool

        if use_linear:
            self.predictor = torch.nn.Linear(emb_dim, num_class)
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, num_class))

    def forward(self, data, epoch=0, eval_random=True):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)
        causaler_output = self.causaler(data)
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        node_cau_num, node_env_num = causaler_output["node_key_num"], causaler_output["node_env_num"]
        edge_cau_num, edge_env_num = causaler_output["edge_key_num"], causaler_output["edge_env_num"]

        cau_node_reg = self.reg_mask_loss(node_cau_num, node_env_num, self.cau_gamma, self.causaler.non_zero_node_ratio)
        cau_edge_reg = self.reg_mask_loss(edge_cau_num, edge_env_num, self.cau_gamma, self.causaler.non_zero_edge_ratio)
        cau_loss_reg = cau_node_reg + cau_edge_reg

        node_env = (1 - node_cau)
        edge_env = (1 - edge_cau)

        h_node_cau = self.graph_backs(x_encode, edge_index, node_cau, edge_cau)
        h_node_env = self.graph_backs(x_encode, edge_index, node_env, edge_env)
        h_graph_cau = self.pool(h_node_cau, batch)
        h_graph_env = self.pool(h_node_env, batch)
        pred_cau = self.predictor(h_graph_cau)
        pred_env = self.predictor(h_graph_env)
        pred_add = self.random_layer(h_graph_cau, h_graph_env, eval_random=eval_random)
        
        """
        edge_cau is very important for our model, we use edge_cau to calculate the causal adjacency matrix,
        later using adjacency matrix to estimate the graphons.
        """
        graphon_loss = 0
        output = {'pred_cau': pred_cau, 
                  'pred_env': pred_env,
                  'pred_add': pred_add,
                  'cau_loss_reg': cau_loss_reg,
                  'graphon_loss': graphon_loss,
                  'causal': causaler_output,
                  'h_graph_env': h_graph_env,
                  'args': self.args}
        
        if self.args.graphon and epoch >= self.args.graphon_pretrain and epoch % self.args.graphon_frequency == 0:
            graphons, ys, envs = estimate_graphon(data, edge_cau.squeeze(), self.N, h_graph_env, self.args.num_env)
            if graphons == None:
                return output
            intra_y = []
            for y in range(len(ys)):
                for env1 in range(len(envs)):
                    for env2 in range(len(envs)):
                        if env1 < env2:
                            graphon1 = graphons['{}{}'.format(y, env1)]
                            graphon2 = graphons['{}{}'.format(y, env2)]
                            intra_y.append(torch.norm(graphon1 - graphon2, p=2))
            output['graphon_loss'] = torch.mean(torch.tensor(intra_y))
        
        return output

    def random_layer(self, xc, xo, eval_random):
        if self.args.random_add == 'shuffle':
            num = xc.shape[0]
            l = [i for i in range(num)]
            if self.args.with_random:
                if eval_random:
                    random.shuffle(l)
            random_idx = torch.tensor(l)
            x = xc[random_idx] + xo
        elif self.args.random_add == 'everyadd':
            x = (xo.unsqueeze(1) + xc.unsqueeze(0)).view(-1, xo.shape[1])
        else:
            assert False
        x_logis = self.predictor(x)
        return x_logis


    def reg_mask_loss(self, key_mask, env_mask, gamma, non_zero_ratio):
        loss_reg =  torch.abs(key_mask / (key_mask + env_mask) - gamma * torch.ones_like(key_mask)).mean()
        loss_reg += (non_zero_ratio - gamma  * torch.ones_like(key_mask)).mean()
        return loss_reg
      