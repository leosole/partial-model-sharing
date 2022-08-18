import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
# https://www.kaggle.com/datasets/shelvigarg/wine-quality-dataset
batch_size = 512
all_col = range(12)
cols = [x for x in all_col]
num_columns = len(cols)
cols_1 = [x for x in all_col if x % 2 == 0 or x > 6]
cols_2 = [x for x in all_col if x % 2 != 0 or x > 6]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 2500, 'initial_train': 0, 'initial_test': 5000, 'end_test': 6497, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 5000, 'initial_train': 2500, 'initial_test': 5000, 'end_test': 6497, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 5000, 'initial_train': 0, 'initial_test':5000, 'end_test': 6497
}
path = 'data/winequalityN.csv'
shared_layers = [512, 256]
ind_layers = [512, 256]
agg_layers = [512, 512, 512]
local_layers = [128, 128, 128]
server_layers = [30,20,10]
dropout = 0.3
weight_decay=0
rounds = 100
epochs = 2
random = 2
freeze = False
resample = False
metrics = 'accuracy'
# metrics = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
# loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss = 'binary_crossentropy'
lr = 0.003
label = -1
targets = 1
activation = 'sigmoid'