# Multi-Grained Invariant Learning for Generalizable Graph Classification
We provide a detailed code for Multi-Grained Invariant Learning for Generalizable Graph Classification.

## Installations
Main packages: PyTorch, Pytorch Geometric, OGB.
```
pytorch==1.10.1
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
ogb==1.3.5
typed-argument-parser==1.7.2
gdown==4.6.0
tensorboard==2.10.1
ruamel-yaml==0.17.21
cilog==1.2.3
munch==2.5.0
rdkit==2020.09.1.0
```

## Preparations
Please download the graph OOD datasets and OGB datasets as described in the original paper. 
Create a folder ```dataset```, and then put the datasets into ```dataset```. Then modify the path by specifying ```--data_dir your/path/dataset```.


## Commands
 We use the NVIDIA GeForce RTX 3090 (24GB GPU) to conduct all our experiments.
 To run the code on CMNIST, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
--dataset cmnist \
--domain color \
--shift covariate \
--emb_dim 300 \
--batch_size 256 \
--epochs 100 \
--trails 10 \
--lr 0.0005 \
--use_linear False \
--lr_decay 100 \
--dropout_rate 0.75 \
--lr_scheduler cos \
--layer 3 \
--l2reg 1e-6 \
--graphon True \
--graphon_pretrain 80 \
--graphon_frequency 10 \
--num_env 5 \
--save_model False \
--cau_gamma 0.6 

```
 

 To run the code on Molbbbp, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
--dataset ogbg-molbbbp \
--domain size \
--emb_dim 32 \
--batch_size 256 \
--epochs 100 \
--trails 10 \
--lr 0.001 \
--use_linear False \
--lr_decay 100 \
--dropout_rate 0.4 \
--lr_scheduler cos \
--layer 3 \
--l2reg 5e-6 \
--graphon True \
--graphon_pretrain 80 \
--graphon_frequency 10 \
--num_env 3 \
--save_model False \
--cau_gamma 0.3 \
--cau 0.5 \
--env 0 \
--inv 0.5 \
--gra 0.5 \
--reg 0.1
```

To run the code on Motif, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
--dataset motif \
--domain basis \
--shift concept \
--emb_dim 300 \
--batch_size 64 \
--epochs 100 \
--trails 10 \
--lr 0.001 \
--use_linear False \
--lr_decay 100 \
--dropout_rate 0.5 \
--lr_scheduler cos \
--layer 2 \
--l2reg 5e-6 \
--graphon True \
--graphon_pretrain 60 \
--graphon_frequency 5 \
--num_env 3 \
--save_model False \
--cau_gamma 0.6 \
--cau 1 \
--env 0 \
--inv 0.001 \
--gra 0.5 \
--reg 0.5
```

 To run the code on Molhiv, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
--dataset hiv \
--domain size \
--shift concept \
--emb_dim 300 \
--batch_size 256 \
--epochs 100 \
--trails 10 \
--lr 0.0001 \
--use_linear False \
--lr_decay 100 \
--dropout_rate 0.1 \
--lr_scheduler cos \
--layer 3 \
--l2reg 1e-6 \
--graphon True \
--graphon_pretrain 70 \
--graphon_frequency 5 \
--num_env 3 \
--save_model False \
--cau_gamma 0.6 \
--cau 1 \
--env 0 \
--inv 0.4 \
--gra 0.4 \
--reg 1
```
