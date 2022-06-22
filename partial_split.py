#%%
import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

import flwr as fl

   
class FraudDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def f1_score(actuals, predictions):
    tp, fp, tn, fn = 0, 0, 0, 0
    for label, pred in zip(actuals, predictions):
        tp += (pred and label)
        fp += (pred and not label)
        tn += (not pred and not label)
        fn += (not pred and label)  
    print(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')      
    return tp / (tp + (fp + fn)/2)

def accuracy(actuals, predictions):
    corrects = actuals and predictions
    return sum(corrects) / len(corrects)

def load_data(path, initial_split, train_split, test_split, initial_test, batch_size, columns=[], random=1, resample=False, drop_columns=[]): 
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=random)
    df = df.drop(columns=drop_columns)
    x_train = df.iloc[initial_split:train_split, 0:-1].values.astype(np.float32)
    y_train = df.iloc[initial_split:train_split, -1].values.astype(np.float32)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    if resample:
        x_train, y_train = SMOTE(random_state=random).fit_resample(x_train, y_train)
    x_test = df.iloc[initial_test:test_split, 0:-1].values.astype(np.float32)
    x_test = sc.transform(x_test)
    y_test = df.iloc[initial_test:test_split, -1].values.astype(np.float32)
    trainset = FraudDataset(x_train, y_train)
    testset = FraudDataset(x_test, y_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    num_examples = {'trainset' : len(trainset), 'testset' : len(testset)}
    n_columns = df.shape[1] - 1
    return trainloader, testloader, num_examples, n_columns 


class Net(nn.Module):
    def __init__(self, sizes, dropout) -> None:
        super(Net, self).__init__()
        self.last = sizes[-1]
        modules = []
        for i in range(len(sizes)-1):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2 or self.last > 2:
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
            else:
                modules.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.sequential(x)

    def predict(self, x):
        pred = torch.F.softmax(self.forward(x))
        return torch.tensor(pred)

class SplitNN(nn.Module):
    def __init__(self, models, ind_columns, shared_columns) -> None:
        super(SplitNN, self).__init__()
        self.ind_model = models['ind_model']
        self.shared_model = models['shared_model']
        self.agg_model = models['agg_model']
        self.ind_columns = ind_columns
        self.shared_columns = shared_columns
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_ind = x[:,self.ind_columns]
        x_shared = x[:,self.shared_columns]
        x_ind = self.ind_model(x_ind)
        x_shared = self.shared_model(x_shared)
        return self.agg_model(torch.cat((x_ind, x_shared), dim=1))
        
    def predict(self, x):
        pred = torch.F.softmax(self.forward(x))
        return torch.tensor(pred)

# %%
