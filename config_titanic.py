import torch
import torch.nn as nn
import numpy as np

batch_size = 64
all_col = range(7)
client1 = {
    'train_split': 350, 'initial_split': 0, 'initial_test': 700, 'test_split': 891, 'batch_size': batch_size, 'columns': [0, 2, 3, 4, 6]
}
client2 = {
    'train_split': 700, 'initial_split': 350, 'initial_test': 700, 'test_split': 891, 'batch_size': batch_size, 'columns': [1, 2, 5, 6]
}
server = {
    'train_split': 700, 'initial_split': 0, 'initial_test': 700, 'test_split': 891, 'batch_size': batch_size, 'columns':[]
}
path = 'data/train_titanic_processed.csv'
shared_layers = [256, 256]
ind_layers = [256, 256]
agg_layers = [512, 2]
local_layers = [512, 512, 2]
server_layers = [512, 512, 2]
dropout = 0.1
weight_decay=0
rounds = 40
epochs = 1
random = 4
freeze = False
resample = False
lr = 0.01