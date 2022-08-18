import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

batch_size = 256
all_col = range(30)
num_columns = len(all_col)
cols_1 = [x for x in all_col if x % 2 == 0 or x > 15]
cols_2 = [x for x in all_col if x % 2 == 0 or x < 15]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 400000, 'initial_train': 0, 'initial_test': 160000, 'end_test': 900000, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 800000, 'initial_train': 400000, 'initial_test': 800000, 'end_test': 900000, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 800000, 'initial_train': 0, 'initial_test':800000, 'end_test': 900000
}
path = 'data/TabularPlaygroundMay2022.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 512]
local_layers = [128, 128, 128]
server_layers = [128, 128, 128]
dropout = 0.3
weight_decay=0
rounds = 10
epochs = 2
random = 2
freeze = False
resample = False
epochs = 30
metrics = 'accuracy'
loss = 'binary_crossentropy'
lr = 0.003