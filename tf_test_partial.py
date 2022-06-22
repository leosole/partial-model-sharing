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

def create_model(num_columns, shared_layers, ind_layers, agg_layers, dropout, lr, split, loss, metrics): 
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

def load_data(path, initial_train, end_train, initial_test, end_test, shared, ind, random=1, label=-1): 
    df = pd.read_csv(path)
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
    # print('init resampling')
    # smote_tomek = SMOTETomek(random_state=random)
    # x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
    # print('end resampling')
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
cols_1 = [x for x in all_col if x % 2 != 0 or x < 200]
cols_2 = [x for x in all_col if x % 2 == 0 or x < 200]
cols_g = [x for x in all_col]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client_1 = {'initial_train':0, 'end_train':250000, 'initial_test':500000, 'end_test':590540, 'shared':shared_columns, 'ind':ind_1}
client_2 = {'initial_train':250000, 'end_train':500000, 'initial_test':500000, 'end_test':590540, 'shared':shared_columns, 'ind':ind_2}
global_client = {'initial_train':0, 'end_train':500000, 'initial_test':500000, 'end_test':590540, 'shared':[], 'ind':cols_g}

num_columns = 212
shared_layers = [256, 128]
ind_layers = [256 ,128]
agg_layers = [256, 256]
dropout = 0.1
lr = 0.0003
batch_size = 2048
loss = 'binary_crossentropy'
metrics = [tfa.metrics.F1Score(num_classes=1, threshold=0.5)]

x_train, y_train, x_test, y_test = load_data(path, **client_1)
split = [len(ind_1), len(shared_columns)]
if sys.argv[1] == '1':
    x_train, y_train, x_test, y_test = load_data(path, **client_1)
    split = [len(ind_1), len(shared_columns)]
    num_columns = len(cols_1)
if sys.argv[1] == '2':
    x_train, y_train, x_test, y_test = load_data(path, **client_2)
    split = [len(ind_2), len(shared_columns)]
    num_columns = len(cols_2)
frauds = sum(y_train)
no_frauds = len(y_train) - frauds
# class_weight = {0:1/no_frauds, 1:1/frauds}
model, shared_model = create_model(num_columns, shared_layers, ind_layers, agg_layers, dropout, lr, split, loss, metrics)
# %%
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return shared_model.get_weights()

    def fit(self, parameters, config):
        shared_model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=2, batch_size=batch_size, validation_data=(x_test, y_test))
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

with open(f'results/tf_partial-ieee-{sys.argv[1]}-{len(shared_columns)}.txt', 'a') as f:
    f.write(f'{metric[0]}\n')