import tensorflow_addons as tfa
import numpy as np

batch_size = 512
all_col = range(253)
num_columns = len(all_col)
cols_1 = [x for x in all_col if x % 2 == 0 or x < 120]
cols_2 = [x for x in all_col if x % 2 != 0 or x < 120]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 4000, 'initial_train': 0, 'initial_test': 8000, 'end_test': 9805, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 8000, 'initial_train': 4000, 'initial_test': 8000, 'end_test': 9805, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 8000, 'initial_train': 0, 'initial_test':8000, 'end_test': 9805
}
path = 'data/train_hcg_processed.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 512]
local_layers = [256, 256, 256]
server_layers = [1024, 1024, 1024]
dropout = 0.4
weight_decay=0
rounds = 100
epochs = 2
random = 11
freeze = False
resample = False
metrics = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
loss = 'binary_crossentropy'
lr = 0.001