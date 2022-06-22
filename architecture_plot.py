#%%
import sys


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchrecorder

import numpy as np
import pandas as pd

import flwr as fl

sys.path.append(".")
import config
#%%
class Net(nn.Module):
    def __init__(self, sizes) -> None:
        super(Net, self).__init__()
        self.last = sizes[-1]
        modules = []
        for i in range(len(sizes)-1):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2 or self.last > 1:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(config.dropout))
            else:
                modules.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.sequential(x)
class SplitNN(nn.Module):
    def __init__(self, models) -> None:
        super(SplitNN, self).__init__()
        self.ind_model = models['ind_model']
        self.shared_model = models['shared_model']
        self.agg_model = models['agg_model']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_ind = x[:,ind_columns]
        x_shared = x[:,shared_columns]
        x_ind = self.ind_model(x_ind)
        x_shared = self.shared_model(x_shared)
        return self.agg_model(torch.cat((x_ind, x_shared), dim=1))

class FraudDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def load_data(path, initial_split, train_split, test_split, initial_test, columns, batch_size=128, label=30): # 'data/creditcard.csv', 2000, 3000, 1:30
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=config.random)
    x_train = df.iloc[initial_split:train_split, 0:-1].values.astype(np.float32)
    y_train = df.iloc[initial_split:train_split, -1].values.astype(np.float32)
    x_test = df.iloc[initial_test:test_split, 0:-1].values.astype(np.float32)
    y_test = df.iloc[initial_test:test_split, -1].values.astype(np.float32)
    trainset = FraudDataset(x_train, y_train)
    testset = FraudDataset(x_test, y_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader, x_train

shared_columns = [col for col in config.client1['columns'] if col in config.client2['columns']]
ind_columns = [col for col in config.client1['columns'] if col not in shared_columns]

trainloader, testloader, x_train = load_data(config.path, **config.client1)

shared_model = Net([len(shared_columns), *config.shared_layers])
ind_model = Net([len(ind_columns), *config.ind_layers])
agg_model = Net([shared_model.last + ind_model.last, *config.agg_layers])
splitNN = SplitNN({'ind_model': ind_model, 'shared_model': shared_model, 'agg_model': agg_model})
opts = torch.optim.Adam(splitNN.parameters(), lr=1e-3, weight_decay=config.weight_decay)
#%%

torchrecorder.render_network(
    splitNN,
    name="Sample Net",
    input_shapes=(2048,324),
    directory="./",
    fmt="svg",
    render_depth=1,
)
# %%
