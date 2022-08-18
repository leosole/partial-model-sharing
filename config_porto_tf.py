from sklearn import metrics
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf

batch_size = 256
all_col = range(207)
num_columns = len(all_col)
cols_1 = [x for x in all_col if x % 2 == 0]
cols_2 = [x for x in all_col if x % 2 != 0]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 250000, 'initial_train': 0, 'initial_test': 500000, 'end_test': 595212, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 500000, 'initial_train': 250000, 'initial_test': 500000, 'end_test': 595212, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 500000, 'initial_train': 0, 'initial_test':500000, 'end_test': 595212
}
path = 'data/train_portoseguro_proc3.csv'
shared_layers = [128, 64]
ind_layers = [128, 64]
agg_layers = [128, 128, 128]
local_layers = [128, 128, 128]
server_layers = [128, 64, 64,32,32,16]
dropout = 0.1
weight_decay=0
label = -1
rounds = 20
epochs = 1
random = 2
freeze = False
resample = False
activation='sigmoid'
targets = 1
metrics = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
metrics=tf.keras.metrics.AUC()
# metrics = 'accuracy'
loss = 'binary_crossentropy'
# loss='categorical_crossentropy'
# cat = True
lr = 0.0003

def gini(actual, pred):
    n = tf.shape(actual)[1]
    indices = tf.reverse(tf.nn.top_k(pred, k=n)[1], axis=[1])[0]
    a_s = tf.gather(tf.transpose(actual), tf.transpose(indices))
    a_c = tf.cumsum(a_s)
    giniSum = tf.reduce_sum(a_c) / tf.reduce_sum(a_s)
    giniSum = tf.subtract(giniSum, tf.divide(tf.cast(n + 1, dtype=tf.float32), tf.constant(2.)))
    return giniSum / tf.cast(n, dtype=tf.float32)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# metrics = gini_normalized