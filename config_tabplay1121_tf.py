import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

batch_size = 2048
all_col = range(67)
cols = [x for x in all_col]
num_columns = len(cols)
# cols_1 = [x for x in all_col if x % 2 != 0 or x > 30]
# cols_2 = [x for x in all_col if x % 2 == 0 or x > 30]
cols_1 = [0, 2, 3, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65]
cols_2 = [0, 1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 250000, 'initial_train': 0, 'initial_test': 500000, 'end_test': 600000, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 500000, 'initial_train': 250000, 'initial_test': 500000, 'end_test': 600000, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 500000, 'initial_train': 0, 'initial_test':500000, 'end_test': 600000
}
path = 'data/TabularPlaygroundNov2021_proc2.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 512]
local_layers = [128, 128, 128]
server_layers = [128,64,32]
dropout = 0.3
weight_decay=0
label = -1
rounds = 10
epochs = 2
random = 2
freeze = False
resample = False
activation='sigmoid'
targets = 1
metrics = 'AUC'
# metrics = tfa.metrics.F1Score(num_classes=7)
loss = 'binary_crossentropy'
lr = 0.0003