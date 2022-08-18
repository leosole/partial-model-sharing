import tensorflow_addons as tfa
import numpy as np

batch_size = 1024
all_col = range(322)
num_columns = len(all_col)
cols_1 = [x for x in all_col if x % 2 != 0 or x < 242]
cols_2 = [x for x in all_col if x % 2 == 0 or x < 242]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {'initial_train':0, 'end_train':250000, 'initial_test':500000, 'end_test':590540, 'shared':shared_columns, 'ind':ind_1}
trad1 = {'initial_train':0, 'end_train':250000, 'initial_test':500000, 'end_test':590540}
client2 = {'initial_train':250000, 'end_train':500000, 'initial_test':500000, 'end_test':590540, 'shared':shared_columns, 'ind':ind_2}
trad2 = {'initial_train':250000, 'end_train':500000, 'initial_test':500000, 'end_test':590540}
global_client = {'initial_train':0, 'end_train':500000, 'initial_test':500000, 'end_test':590540}
path = 'data/train_transaction_processed.csv'
shared_layers = [256, 128]
ind_layers = [256, 128]
agg_layers = [256, 256]
local_layers = [256, 256, 256]
server_layers = [256, 256, 256]
dropout = 0.1
weight_decay=0
rounds = 30
epochs = 2
random = 22
freeze = False
resample = False
metrics = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
loss = 'binary_crossentropy'
lr = 0.0003
label = -1
targets = 1
activation = 'sigmoid'