from sklearn import metrics
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf

batch_size = 256
all_col = range(39)
num_columns = len(all_col)
cols = [x for x in all_col]
# cols_1 = [x for x in all_col if x % 2 != 0 or x > 21 or x < 10]
cols_1 = [x for x in all_col if x < 37]
# cols = cols_1
num_columns = len(cols)
# cols_2 = [x for x in all_col if x % 2 == 0 or x > 21 or x < 10]
cols_2 = [x for x in all_col if x > 7]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 25000, 'initial_train': 0, 'initial_test': 50000, 'end_test': 61878, 'shared':shared_columns, 'ind':ind_1
}
trad1 = {
    'end_train': 25000, 'initial_train': 0, 'initial_test': 50000, 'end_test': 61878
}
client2 = {
    'end_train': 50000, 'initial_train': 25000, 'initial_test': 50000, 'end_test': 61878, 'shared':shared_columns, 'ind':ind_2
}
trad2 = {
    'end_train': 50000, 'initial_train': 25000, 'initial_test': 50000, 'end_test': 61878
}
global_client = {
    'end_train': 50000, 'initial_train': 0, 'initial_test':50000, 'end_test': 61878
}
path = 'data/ottoGroupProduct_proc.csv'
shared_layers = [128, 64]
ind_layers = [128, 64]
agg_layers = [128, 64]
local_layers = [128, 128, 64]
server_layers = [128, 64]
dropout = 0.1
weight_decay=0
label = [-1,-2,-3,-4,-5,-6,-7,-8,-9]
rounds = 50
epochs = 1
random = 2
freeze = False
resample = False
activation='softmax'
targets = 9
# metrics = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
# metrics=tf.keras.metrics.AUC()
metrics = 'accuracy'
# loss = 'binary_crossentropy'
loss='categorical_crossentropy'
# cat = True
lr = 0.0003
