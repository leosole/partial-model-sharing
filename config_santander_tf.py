import tensorflow_addons as tfa
import numpy as np

batch_size = 2048
all_col = range(200)
num_columns = len(all_col)
cols_1 = [x for x in all_col if x % 2 == 0 or x < 100]
cols_2 = [x for x in all_col if x % 2 != 0 or x < 100]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 80000, 'initial_train': 0, 'initial_test': 160000, 'end_test': 200000, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 160000, 'initial_train': 80000, 'initial_test': 160000, 'end_test': 200000, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 160000, 'initial_train': 0, 'initial_test':160000, 'end_test': 200000
}
path = 'data/train_santander_processed.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 512]
local_layers = [512, 512, 512]
server_layers = [512, 265, 128]
dropout = 0.3
weight_decay=0
rounds = 50
epochs = 1
random = 2
freeze = False
resample = False
epochs = 30
metrics = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
loss = 'binary_crossentropy'
lr = 0.03