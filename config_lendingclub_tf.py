import tensorflow_addons as tfa
import numpy as np

batch_size = 2048
all_col = range(74)
cols = [x for x in all_col if x > 50]
num_columns = len(cols)
cols_1 = [x for x in cols if x % 2 == 0 or x < 60]
cols_2 = [x for x in cols if x % 2 != 0 or x < 60]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 150000, 'initial_train': 0, 'initial_test': 300000, 'end_test': 395219, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 300000, 'initial_train': 150000, 'initial_test': 300000, 'end_test': 395219, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 300000, 'initial_train': 0, 'initial_test':300000, 'end_test': 395219
}
path = 'data/lending_club_loan_processed.csv'
shared_layers = [128, 64]
ind_layers = [128, 64]
agg_layers = [128, 128, 128]
local_layers = [128, 128, 128]
server_layers = [128, 128, 128]
dropout = 0.3
weight_decay=0
rounds = 20
epochs = 1
random = 2
freeze = False
resample = False
metrics = 'accuracy'
loss = 'binary_crossentropy'
lr = 0.003