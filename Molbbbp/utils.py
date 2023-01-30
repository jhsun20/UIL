import torch
import numpy as np
import random
import pdb
from torch_geometric.transforms import BaseTransform
from ogb.graphproppred import PygGraphPropPredDataset
def get_info_dataset(args, dataset, split_idx):

    total = []
    for mode in ['train', 'valid', 'test']:
        mode_max_node = 0
        mode_min_node = 9999
        mode_avg_node = 0
        mode_tot_node = 0.0

        dataset_name = dataset[split_idx[mode]]
        mode_num_graphs = len(dataset_name)
        for data in dataset_name:
            num_node = data.num_nodes
            mode_tot_node += num_node
            if num_node > mode_max_node:
                mode_max_node = num_node
            if num_node < mode_min_node:
                mode_min_node = num_node
        print("{} {:<5} | Graphs num:{:<5} | Node num max:{:<4}, min:{:<4}, avg:{:.2f}"
            .format(args.dataset, mode, mode_num_graphs,
                                        mode_max_node,
                                        mode_min_node, 
                                        mode_tot_node / mode_num_graphs))
        total.append(mode_num_graphs)
    all_graph_num = sum(total)
    print("train:{:.2f}%, val:{:.2f}%, test:{:.2f}%"
        .format(float(total[0]) * 100 / all_graph_num, 
                float(total[1]) * 100 / all_graph_num, 
                float(total[2]) * 100 / all_graph_num))

def size_split_idx(dataset, mode):

    num_graphs = len(dataset)
    num_val   = int(0.1 * num_graphs)
    num_test  = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []
    valtest_list = []

    for data in dataset:
        num_node_list.append(data.num_nodes)

    sort_list = np.argsort(num_node_list)

    if mode == 'ls':
        train_idx = sort_list[2 * num_val:]
        valid_test_idx = sort_list[:2 * num_val]
    else:
        train_idx = sort_list[:-2 * num_val]
        valid_test_idx = sort_list[-2 * num_val:]
    random.shuffle(valid_test_idx)
    valid_idx = valid_test_idx[:num_val]
    test_idx = valid_test_idx[num_val:]

    split_idx = {'train': torch.tensor(train_idx, dtype = torch.long), 
                 'valid': torch.tensor(valid_idx, dtype = torch.long), 
                 'test': torch.tensor(test_idx, dtype = torch.long)}
    return split_idx
    
 

class ToEnvs(BaseTransform):
    
    def __init__(self, envs=10):
        self.envs = envs

    def __call__(self, data):

        data.env_id = torch.randint(0, self.envs, (1,))
        return data

############################################################

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.CEX = False

def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)

def init_weights(net, init_type='orthogonal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def load_data(args):
    dataset = PygGraphPropPredDataset('ogbg-molbbbp')
    in_dim = 9
    num_class = dataset.num_tasks
    eval_metric = 'rocauc'
    num_layer = 3
    cri = torch.nn.BCEWithLogitsLoss()
    eval_name = 'ogbg-molbbbp'
    test_batch_size = args.batch_size    
    return dataset, in_dim, num_class, num_layer, eval_metric, cri, eval_name, test_batch_size
