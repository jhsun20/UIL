import copy
import pdb
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
# from skimage.restoration import denoise_tv_chambolle
from torch_geometric.data import Data
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj
import time
import pdb

def flip(x, dim):
    """
    This function is used for inverse the tensor x.
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def align_graphs(graphs, padding: bool = False, N: int = None):
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * torch.sum(graphs[i], dim=0) + 0.5 * torch.sum(graphs[i], dim=1)
        node_degree = node_degree / torch.sum(node_degree)
        idx = torch.argsort(node_degree)  # ascending
        idx = flip(idx, dim=0)  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        # sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = graphs[i]
        # TODO: check above line is correct or right!
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = torch.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = torch.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph.unsqueeze(dim=0))
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph.unsqueeze(dim=0))

    if N:
        aligned_graphs = [aligned_graph[:, :N, :N] for aligned_graph in aligned_graphs]
        normalized_node_degrees = [normalized_node_degree[:N, :] for normalized_node_degree in normalized_node_degrees]
            
    aligned_graphs = torch.cat(aligned_graphs, dim=0)
    return aligned_graphs, normalized_node_degrees, max_num, min_num


def gra2graphon(aligned_graphs, threshold: float = 2.02):
    num_graphs = aligned_graphs.shape[0]
    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)
    return sum_graph


from sklearn.cluster import KMeans
def estimate_graphon(data, edge_att, N, h_graph_env, num_env):
    batch_size = data.y.shape[0]
    device = data.y.device
    idx = torch.tensor(np.arange(batch_size)).to(device)
    ys = torch.unique(data.y, sorted=True)
    kmeans = KMeans(n_clusters=num_env, random_state=0)
    envs = torch.tensor(list(np.arange(num_env))).to(device)
    y_env_graphs = {}
    graphons = {}
    for y in ys:
        for env in envs:
            y_env_graphs['{}{}'.format(int(y), int(env))] = []
        
    edge_index = data.edge_index
    adj_cau = to_dense_adj(edge_index=edge_index, edge_attr=edge_att)[0]
    for y in ys:
        h_graph_env_y = h_graph_env[data.y.squeeze()==y]
        idx_y = idx[data.y.squeeze()==y]
        if h_graph_env_y.shape[0] < num_env:
            return None, None, None
        kmeans.fit(h_graph_env_y.to('cpu').detach().numpy())
        for i in range(len(idx_y)):
            y_env_graphs['{}{}'.format(int(data[idx_y[i]].y), int(kmeans.labels_[i]))].append(adj_cau[data.ptr[idx_y[i]]:data.ptr[idx_y[i]+1], data.ptr[idx_y[i]]:data.ptr[idx_y[i]+1]])
    for y in ys:
        for env in envs:
            graphs = y_env_graphs['{}{}'.format(int(y), int(env))]
            # print("y:{}, env:{}, num_graphs:{}".format(y, env, len(graphs)))
            aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True, N=N)
            graphon = gra2graphon(aligned_graphs, threshold=0.2)
            graphons['{}{}'.format(int(y),int(env))] = graphon
    # for y, env, graphon in graphons:
        # print("graphon info: class_label: {}, env_label: {}, mean: {}, shape, {}".format(y, env, graphon.mean(), graphon.shape)) 
    return graphons, ys, envs

def stat_graph(dataset):
    
    num_total_nodes = []
    num_total_edges = []
    
    for graph in dataset:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(graph.edge_index.shape[1] )
    avg_num_nodes = sum( num_total_nodes ) / len(dataset)
    avg_num_edges = sum( num_total_edges ) / len(dataset) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median( num_total_nodes ) 
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density

def draw_graphon(data, edge_att, N, h_graph_env, num_env):
    """
    Here we use the ground truth env label to draw graphons
    """
    batch_size = data.y.shape[0]
    device = data.y.device
    idx = torch.tensor(np.arange(batch_size)).to(device)
    ys = torch.unique(data.y, sorted=True)
    envs = torch.unique(data.env_id, sorted=True)
    y_env_graphs = {}
    graphons = {}
    for y in ys:
        for env in envs:
            y_env_graphs['{}{}'.format(y, env)] = []
        
    edge_index = data.edge_index
    adj_cau = to_dense_adj(edge_index=edge_index, edge_attr=edge_att)[0]
    for y in ys:
        idx_y = idx[data.y==y]
        for i in range(len(idx_y)):
            y_env_graphs['{}{}'.format(int(data[idx_y[i]].y), int(data[idx_y[i]].env_id))].append(adj_cau[data.ptr[idx_y[i]]:data.ptr[idx_y[i]+1], data.ptr[idx_y[i]]:data.ptr[idx_y[i]+1]])
    for y in ys:
        for env in envs:
            graphs = y_env_graphs['{}{}'.format(y, env)]
            # print("y:{}, env:{}, num_graphs:{}".format(y, env, len(graphs)))
            aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True, N=N)
            graphon = gra2graphon(aligned_graphs, threshold=0.2)
            graphons['{}{}'.format(y,env)] = graphon
    # for y, env, graphon in graphons:
        # print("graphon info: class_label: {}, env_label: {}, mean: {}, shape, {}".format(y, env, graphon.mean(), graphon.shape)) 
    return graphons, ys, envs