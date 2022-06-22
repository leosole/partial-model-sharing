import torch
import torch.nn as nn
import numpy as np

batch_size = 2048
all_col = range(324)
client1 = {
    'train_split': 250000, 'initial_split': 0, 'initial_test': 250000, 'test_split': 500000, 'batch_size': batch_size, 'columns': [x for x in all_col if x % 2 == 0 or x < 100],
}
client2 = {
    'train_split': 500000, 'initial_split': 250000, 'initial_test': 500000, 'test_split': 590540, 'batch_size': batch_size, 'columns': [x for x in all_col if x % 2 != 0 or x < 100]
}
server = {
    'train_split': 500000, 'initial_split': 0, 'initial_test': 500000, 'test_split': 590540, 'batch_size': batch_size
}
path = 'data/train_transaction_processed.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 2]
local_layers = [512, 512, 512, 2]
server_layers = [512, 512, 512, 2]
dropout = 0.4
weight_decay=0
rounds = 30
epochs = 1
random = 2
freeze = False
resample = False
lr = 0.003