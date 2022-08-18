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
from importlib import import_module
from config_ieee_tf import *
#%%

def create_model(num_columns, shared_layers, ind_layers, agg_layers, dropout, lr, split, loss, metrics, targets=1, activation='sigmoid'): 
    in_all = tf.keras.layers.Input(shape=(num_columns, ))
    in_ind, in_shared = tf.split(in_all, split, 1)
    x_ind = tf.keras.layers.BatchNormalization()(in_ind)
    x_shared = tf.keras.layers.BatchNormalization()(in_shared)

    for i in shared_layers:
        x_shared = tf.keras.layers.Dense(i, activation='relu')(x_shared)
        x_shared = tf.keras.layers.Dropout(dropout)(x_shared)
        x_shared = tf.keras.layers.BatchNormalization()(x_shared)

    for i in ind_layers:
        x_ind = tf.keras.layers.Dense(i, activation='relu')(x_ind)
        x_ind = tf.keras.layers.Dropout(dropout)(x_ind)
        x_ind = tf.keras.layers.BatchNormalization()(x_ind)

    in_agg = tf.keras.layers.concatenate([x_ind, x_shared])
    x_agg = tf.keras.layers.BatchNormalization()(in_agg)

    for i in agg_layers:
        x_agg = tf.keras.layers.Dense(i, activation='relu')(x_agg)
        x_agg = tf.keras.layers.Dropout(dropout)(x_agg)
        x_agg = tf.keras.layers.BatchNormalization()(x_agg)
    out = tf.keras.layers.Dense(targets, activation=activation)(x_agg)
        
    shared_model = tf.keras.Model(in_shared, x_shared)
    model = tf.keras.models.Model(in_all, out)
    model.compile(optimizer=tf.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model, shared_model

def load_data(path, initial_train, end_train, initial_test, end_test, shared, ind, random=1, label=-1): 
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    df = df.sample(frac=1, random_state=random)
    columns = shared + ind
    shared_ids = [columns.index(i) for i in shared]
    ind_ids = [columns.index(i) for i in ind]
    x_train = df.iloc[initial_train:end_train, columns]#.values.astype(np.float32)
    y_train = df.iloc[initial_train:end_train, label].values.astype(np.float32)
    x_test = df.iloc[initial_test:end_test, columns]#.values.astype(np.float32)
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

x_train, y_train, x_test, y_test = load_data(path, **client1, label=label)
split = [len(ind_1), len(shared_columns)]
num_columns = len(cols_1)
if sys.argv[1] == '1':
    x_train, y_train, x_test, y_test = load_data(path, **client1, label=label)
    split = [len(ind_1), len(shared_columns)]
    num_columns = len(cols_1)
if sys.argv[1] == '2':
    x_train, y_train, x_test, y_test = load_data(path, **client2, label=label)
    split = [len(ind_2), len(shared_columns)]
    num_columns = len(cols_2)

model, shared_model = create_model(num_columns, shared_layers, ind_layers, agg_layers, dropout, lr, split, loss, metrics, targets, activation)
# %%
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return shared_model.get_weights()

    def fit(self, parameters, config):
        shared_model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
        return shared_model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        shared_model.set_weights(parameters)
        loss, metric = model.evaluate(x_test, y_test)
        metric = float(metric)
        print(f'TEST: loss: {loss:.4f}, metric: {metric:.4f}')
        return loss, len(x_test), {"metric": metric}

print('start federated')
fl.client.start_numpy_client("[::]:8080", client=FlowerClient())

loss, metric = model.evaluate(x_test, y_test)
print(metric)
with open(f'results/tf_partial_{sys.argv[1]}-ieee-{len(shared_columns)}.txt', 'a') as f:
    f.write(f'{metric[0]}\n')