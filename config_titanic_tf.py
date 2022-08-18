import tensorflow_addons as tfa
import numpy as np

batch_size = 64
all_col = range(7)
num_columns = len(all_col)
cols_1 = [0, 2, 4, 5, 6]
cols_2 = [1, 3, 4, 5, 6]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 350, 'initial_train': 0, 'initial_test': 700, 'end_test': 891, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 700, 'initial_train': 350, 'initial_test': 700, 'end_test': 891, 'shared':shared_columns, 'ind':ind_2
}
global_client = {'initial_train':0, 'end_train':700, 'initial_test':700, 'end_test':891}
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
metrics = [tfa.metrics.F1Score(num_classes=1, threshold=0.5), 'accuracy']
loss = 'binary_crossentropy'