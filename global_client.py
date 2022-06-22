#%%
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(".")
from importlib import import_module
# if len(sys.argv) > 1:
#     config = import_module(sys.argv[1])
# else: 
#     import config
config = import_module('config_ieee')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FraudDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def load_data(path, initial_split, train_split, test_split, columns=[], batch_size=128): 
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=config.random)
    y_train = df.iloc[initial_split:train_split, -1].values.astype(np.float32)
    df = df.drop(columns=columns)
    x_train = df.iloc[initial_split:train_split, 0:-1].values.astype(np.float32)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    if config.resample:
        x_train, y_train = SMOTE(random_state=config.random).fit_resample(x_train, y_train)
    x_test = df.iloc[train_split:test_split, 0:-1].values.astype(np.float32)
    x_test = sc.transform(x_test)
    y_test = df.iloc[train_split:test_split, -1].values.astype(np.float32)
    trainset = FraudDataset(x_train, y_train)
    testset = FraudDataset(x_test, y_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size)
    n_columns = df.shape[1] - 1
    return trainloader, testloader, n_columns

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

losses = []
metrics = []

def train(model, trainloader, epochs):
    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for i in range(epochs):
        actual = []
        predictions = []
        tp, fp, tn, fn = 0, 0, 0, 0
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            outputs = model(x)
            outputs = outputs.view(-1)
            loss = criterion(outputs, y)
            loss.backward()
            opt.step()
            preds = np.round_(outputs.detach().numpy())
            actual.extend(y)
            predictions.extend(preds)
            for lab, pred in zip(y, preds):
                # Collect statistics
                tp += (pred and lab)
                fp += (pred and not lab)
                tn += (not pred and not lab)
                fn += (not pred and lab)        
        f1_score = tp / (tp + (fp + fn)/2)
        metrics.append(f1_score)
        gini = gini_normalized(actual, predictions)
        print(f'\rTRAIN [{i/epochs*100:.2f}%] tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn} | GINI: {gini:.4f}    F1 score: {f1_score:.4f}    Loss: {loss:.4f}', end='')
        losses.append(loss.item())

def test(model, testloader):
    criterion = nn.BCELoss()
    loss = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    actual = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            outputs = outputs.view(-1)
            loss += criterion(outputs, y).item()

            preds = np.round_(outputs.numpy())
            actual.extend(y)
            predictions.extend(preds)
            for lab, pred in zip(y, preds):
                # Collect statistics
                tp += (pred and lab)
                fp += (pred and not lab)
                tn += (not pred and not lab)
                fn += (not pred and lab)
    f1_score = tp / (tp + (fp + fn)/2)
    gini = gini_normalized(actual, predictions)
    print(f'TEST tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn} | GINI: {gini:.4f}    F1 score: {f1_score:.4f}    Loss: {loss:.4f}')
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
                modules.append(nn.Dropout(config.dropout))
            else:
                modules.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.sequential(x)
    
    def predict(self, x):
        pred = torch.F.softmax(self.forward(x))
        return torch.tensor(pred)

trainloader, testloader, n_columns = load_data(config.path, **config.server)
print(n_columns)
model = Net([n_columns, *config.server_layers])
print('begin training')
train(model, trainloader, config.rounds)
print()
_, f1_score =test(model, testloader)
print('Client DONE!')
if len(sys.argv) > 1:
    with open(f'results/global_client-{sys.argv[1]}.txt', 'a') as f:
        f.write(f'{f1_score}\n')
else:
    with open(f'results/global_client-.txt', 'a') as f:
        f.write(f'{f1_score}\n')
# %%
