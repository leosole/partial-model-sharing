#%%
import tensorflow as tf
import tensorflow_addons as tfa
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import sys
#%%

def create_model(num_columns, layers, dropout, lr, loss, metrics): 
    in_all = tf.keras.layers.Input(shape=(num_columns, ))
    x = tf.keras.layers.BatchNormalization()(in_all)

    for i in layers:
        x = tf.keras.layers.Dense(i, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
    model = tf.keras.models.Model(in_all, out)
    model.compile(optimizer=tf.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model

def load_partial_data(path, initial_train, end_train, initial_test, end_test, shared, ind, random=1, label=-1): 
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=random)
    columns = shared + ind
    shared_ids = [columns.index(i) for i in shared]
    ind_ids = [columns.index(i) for i in ind]
    x_train = df.iloc[initial_train:end_train, columns]
    y_train = df.iloc[initial_train:end_train, label].values.astype(np.float32)
    x_test = df.iloc[initial_test:end_test, columns]
    y_test = df.iloc[initial_test:end_test, label].values.astype(np.float32)
    for feat in x_train.columns.values:
        ss = StandardScaler()
        x_train[feat] = ss.fit_transform(x_train[feat].values.reshape(-1,1))
        x_test[feat] = ss.transform(x_test[feat].values.reshape(-1,1))
    x_train = x_train.iloc[:, :].values.astype(np.float32)
    x_test = x_test.iloc[:, :].values.astype(np.float32)
    x_shared_train = x_train[:, shared_ids]
    x_ind_train = x_train[:, ind_ids]
    x_train = np.concatenate([x_ind_train, x_shared_train], axis=1)
    x_shared_test = x_test[:, shared_ids]
    x_ind_test = x_test[:, ind_ids]
    x_test = np.concatenate([x_ind_test, x_shared_test], axis=1)
    return x_train, y_train, x_test, y_test
#%%
path = 'data/train_transaction_processed.csv'
all_col = range(324)
global_client = {'initial_train':0, 'end_train':500000, 'initial_test':500000, 'end_test':590540}

cols_1 = [x for x in all_col if x % 2 != 0 or x < 200]
cols_2 = [x for x in all_col if x % 2 == 0 or x < 200]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client_1 = {'initial_train':0, 'end_train':250000, 'initial_test':500000, 'end_test':590540, 'shared':shared_columns, 'ind':ind_1}
client_2 = {'initial_train':250000, 'end_train':500000, 'initial_test':500000, 'end_test':590540, 'shared':shared_columns, 'ind':ind_2}

num_columns = 212
layers = [256, 256, 256]
dropout = 0.1
lr = 0.0003
batch_size = 2048
loss = 'binary_crossentropy'
metrics = [tfa.metrics.F1Score(num_classes=1, threshold=0.5)]

if sys.argv[1] == '1':
    x_train, y_train, x_test, y_test = load_partial_data(path, **client_1)
    num_columns = len(cols_1)
if sys.argv[1] == '2':
    x_train, y_train, x_test, y_test = load_partial_data(path, **client_2)
    num_columns = len(cols_2)
frauds = sum(y_train)
no_frauds = len(y_train) - frauds
#%%
# class_weight = {0:1/no_frauds, 1:1/frauds}
model = create_model(num_columns, layers, dropout, lr, loss, metrics)

model.fit(x_train, y_train, epochs=30, batch_size=batch_size, validation_data=(x_test, y_test))

loss, metric = model.evaluate(x_test, y_test)
with open(f'results/tf_local-ieee-{sys.argv[1]}-{len(shared_columns)}.txt', 'a') as f:
    f.write(f'{metric[0]}\n')
# %%
