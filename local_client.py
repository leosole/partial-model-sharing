#%%
import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

import flwr as fl

sys.path.append(".")
import config

if len(sys.argv) < 1:
    print('Error: client number needed')
    exit()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    x_train = df.iloc[initial_split:train_split, 0:label].values.astype(np.float32)
    y_train = df.iloc[initial_split:train_split, label].values.astype(np.float32)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = df.iloc[initial_test:test_split, 0:label].values.astype(np.float32)
    x_test = sc.transform(x_test)
    y_test = df.iloc[initial_test:test_split, label].values.astype(np.float32)
    trainset = FraudDataset(x_train, y_train)
    testset = FraudDataset(x_test, y_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader

if sys.argv[1] == '1':
    trainloader, testloader = load_data('data/creditcard.csv', **config.client1)
    columns = config.client1['columns']

if sys.argv[1] == '2':
    trainloader, testloader = load_data('data/creditcard.csv', **config.client2)
    columns = config.client2['columns']

def train(model, trainloader, epochs):
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.03)
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            outputs = model(x[:,columns])
            outputs = outputs.view(-1)
            loss = criterion(outputs, y)
            loss.backward()
            opt.step()

def test(model, testloader):
    criterion = nn.BCELoss()
    loss = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x[:,columns])
            outputs = outputs.view(-1)
            loss += criterion(outputs, y).item()

            preds = np.round_(outputs.numpy())
            for lab, pred in zip(y, preds):
                # Collect statistics
                tp += (pred and lab)
                fp += (pred and not lab)
                tn += (not pred and not lab)
                fn += (not pred and lab)
    f1_score = tp / (tp + (fp + fn)/2)
    print(f'TEST tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn} | F1 score: {f1_score:.4f} \t Loss: {loss:.4f}')
    return loss, f1_score

class Net(nn.Module):
    def __init__(self, sizes) -> None:
        super(Net, self).__init__()
        self.last = sizes[-1]
        modules = []
        for i in range(len(sizes)-1):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2 or self.last > 1:
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.sequential(x)

model = Net([len(columns), *config.local_layers])
print('begin training')
train(model, trainloader, config.rounds)
test(model, testloader)
print('Client DONE!')