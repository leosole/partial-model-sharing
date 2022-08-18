import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

batch_size = 256
all_col = range(54)
cols = [x for x in all_col]
num_columns = len(cols)
cols_1 = [x for x in all_col if x % 2 == 0 or x > 26]
cols_2 = [x for x in all_col if x % 2 != 0 or x > 26]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 1500000, 'initial_train': 0, 'initial_test': 3000000, 'end_test': 4000000, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 3000000, 'initial_train': 1500000, 'initial_test': 3000000, 'end_test': 4000000, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 3000000, 'initial_train': 0, 'initial_test':3000000, 'end_test': 4000000
}
path = 'data/TabularPlaygroundDec2021.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 512]
local_layers = [128, 128, 128]
server_layers = [128, 128, 128]
dropout = 0.3
weight_decay=0
label = [54,55,56,57,58,59,60]
rounds = 10
epochs = 2
random = 2
freeze = False
resample = False
activation='relu'
targets = 7
# metrics = 'accuracy'
metrics = 'categorical_accuracy'
# metrics = tfa.metrics.F1Score(num_classes=7)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
lr = 0.003