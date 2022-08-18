#%%
import tensorflow as tf
import tensorflow_addons as tfa
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import sys
sys.path.append(".")
from config_ieee_tf import *
#%%

def create_model(num_columns, layers, dropout, lr, loss, metrics, targets=1, activation='sigmoid'): 
    in_all = tf.keras.layers.Input(shape=(num_columns, ))
    x = tf.keras.layers.BatchNormalization()(in_all)

    for i in layers:
        x = tf.keras.layers.Dense(i, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

    out = tf.keras.layers.Dense(targets, activation=activation)(x)
        
    model = tf.keras.models.Model(in_all, out)
    model.compile(optimizer=tf.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model

def load_partial_data(path, initial_train, end_train, initial_test, end_test, shared, ind, label=-1): 
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
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
    if 'cat' in globals():
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_partial_data(path, **client2, label=label)
num_columns = len(cols_2)
if sys.argv[1] == '1':
    x_train, y_train, x_test, y_test = load_partial_data(path, **client1, label=label)
    num_columns = len(cols_1)
if sys.argv[1] == '2':
    x_train, y_train, x_test, y_test = load_partial_data(path, **client2, label=label)
    num_columns = len(cols_2)
frauds = sum(y_train)
no_frauds = len(y_train) - frauds
#%%
# class_weight = {0:1/no_frauds, 1:1/frauds}
model = create_model(num_columns, local_layers, dropout, lr, loss, metrics, targets, activation)

model.fit(x_train, y_train, epochs=rounds, batch_size=batch_size, validation_data=(x_test, y_test))

loss, metric = model.evaluate(x_test, y_test)
# print(metric)
with open(f'results/tf_local_{sys.argv[1]}-ieee-{len(shared_columns)}.txt', 'a') as f:
    f.write(f'{metric[0]}\n')
# %%
