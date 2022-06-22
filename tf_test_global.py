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

def load_data(path, initial_train, end_train, initial_test, end_test, random=1, label=-1): 
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=random)
    x_train = df.iloc[initial_train:end_train, :-1]#.values.astype(np.float32)
    y_train = df.iloc[initial_train:end_train, label].values.astype(np.float32)
    x_test = df.iloc[initial_test:end_test, :-1]#.values.astype(np.float32)
    y_test = df.iloc[initial_test:end_test, label].values.astype(np.float32)
    for feat in x_train.columns.values:
        ss = StandardScaler()
        x_train[feat] = ss.fit_transform(x_train[feat].values.reshape(-1,1))
        x_test[feat] = ss.transform(x_test[feat].values.reshape(-1,1))
    x_train = x_train.iloc[:, :].values.astype(np.float32)
    x_test = x_test.iloc[:, :].values.astype(np.float32)
    # print('init resampling')
    # smote_tomek = SMOTETomek(random_state=random)
    # x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
    # print('end resampling')
    return x_train, y_train, x_test, y_test
#%%
path = 'data/train_transaction_processed.csv'
all_col = range(324)
global_client = {'initial_train':0, 'end_train':500000, 'initial_test':500000, 'end_test':590540}

num_columns = 324
layers = [256, 256, 256]
dropout = 0.1
lr = 0.0003
batch_size = 1024
loss = 'binary_crossentropy'
metrics = [tfa.metrics.F1Score(num_classes=1, threshold=0.5)]

x_train, y_train, x_test, y_test = load_data(path, **global_client)
frauds = sum(y_train)
no_frauds = len(y_train) - frauds
#%%
# class_weight = {0:1/no_frauds, 1:1/frauds}
model = create_model(num_columns, layers, dropout, lr, loss, metrics)

model.fit(x_train, y_train, epochs=30, batch_size=batch_size, validation_data=(x_test, y_test))

loss, metric = model.evaluate(x_test, y_test)
with open(f'results/tf_global-ieee.txt', 'a') as f:
    f.write(f'{metric[0]}\n')
