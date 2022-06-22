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

def create_split_model(num_columns, shared_layers, ind_layers, agg_layers, dropout, lr, split, loss, metrics): 
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
    x_agg = tf.keras.layers.Dense(1, activation='sigmoid')(x_agg)
        
    shared_model = tf.keras.Model(in_shared, x_shared)
    model = tf.keras.models.Model(in_all, x_agg)
    model.compile(optimizer=tf.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model, shared_model

def create_local_model(num_columns, layers, dropout, lr, loss, metrics): 
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

def load_data(path, initial_train, end_train, initial_test, end_test, random=1, label=-1): 
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=random)
    x_train = df.iloc[initial_train:end_train, :-1]
    y_train = df.iloc[initial_train:end_train, label].values.astype(np.float32)
    x_test = df.iloc[initial_test:end_test, :-1]
    y_test = df.iloc[initial_test:end_test, label].values.astype(np.float32)
    for feat in x_train.columns.values:
        ss = StandardScaler()
        x_train[feat] = ss.fit_transform(x_train[feat].values.reshape(-1,1))
        x_test[feat] = ss.transform(x_test[feat].values.reshape(-1,1))
    x_train = x_train.iloc[:, :].values.astype(np.float32)
    x_test = x_test.iloc[:, :].values.astype(np.float32)
    return x_train, y_train, x_test, y_test

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
