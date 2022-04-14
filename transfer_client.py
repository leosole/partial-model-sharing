#%%
import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import flwr as fl

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
class FraudDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def load_data(path, initial_split, train_split, test_split, columns, batch_size=128, label=30): # 'data/creditcard.csv', 2000, 3000, 1:30
    df = pd.read_csv(path)
    x_train = df.iloc[initial_split:train_split, 0:label].values.astype(np.float32)
    y_train = df.iloc[initial_split:train_split, label].values.astype(np.float32)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = df.iloc[train_split:test_split, 0:label].values.astype(np.float32)
    x_test = sc.transform(x_test)
    y_test = df.iloc[train_split:test_split, label].values.astype(np.float32)
    trainset = FraudDataset(x_train, y_train)
    testset = FraudDataset(x_test, y_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    num_examples = {'trainset' : len(trainset), 'testset' : len(testset)}
    return trainloader, testloader, num_examples

# %%
client1_args = {
    'train_split': 100000, 'initial_split': 0, 'test_split': 120000, 'batch_size': 128, 'label': 30, 'columns': [*range(1,30)]
}
client2_args = {
    'train_split': 220000, 'initial_split': 120000, 'test_split': 240000, 'batch_size': 128, 'label': 30, 'columns': [*range(1,30)]
}

# %%
# trainloader, testloader, num_examples = load_data('data/creditcard.csv', **client1_args)
shared_columns = [col for col in client1_args['columns'] if col in client2_args['columns']]
# ind_columns = [col for col in client1_args['columns'] if col not in shared_columns]

if sys.argv[1] == '1':
    trainloader, testloader, num_examples = load_data('data/creditcard.csv', **client1_args)
    # ind_columns = [col for col in client1_args['columns'] if col not in shared_columns]

if sys.argv[1] == '2':
    trainloader, testloader, num_examples = load_data('data/creditcard.csv', **client2_args)
    # ind_columns = [col for col in client2_args['columns'] if col not in shared_columns]
# %%

def fed_train(shared_model, shared_opt, trainloader, epochs):
    criterion = nn.BCELoss()
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            shared_opt.zero_grad()
            outputs = shared_model(x[:,shared_columns])
            outputs = outputs.view(-1)
            loss = criterion(outputs, y)
            loss.backward()
            shared_opt.step()

def fed_test(shared_model, testloader):
    criterion = nn.BCELoss()
    loss = 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = shared_model(x[:,shared_columns])
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
    print(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')
    print(f'F1 score: {f1_score} \t Loss: {loss}')  
    return loss, f1_score

class SplitNN(nn.Module):
    def __init__(self, sizes) -> None:
        super(SplitNN, self).__init__()
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        self.lin1 = nn.Linear(sizes[0], sizes[1])
        self.lin2 = nn.Linear(sizes[1], sizes[2])
        self.lin3 = nn.Linear(sizes[2], sizes[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return torch.sigmoid(x) 

# %%
print(f'number of shared features: {len(shared_columns)}')
print(f'number of individual features: {len(ind_columns)}')
shared_model = SplitNN([len(shared_columns), 96, 96, 1])
shared_opt = torch.optim.SGD(shared_model.parameters(), lr=0.03, momentum=0.9)
# ind_opt = torch.optim.SGD(ind_model.parameters(), lr=0.001, momentum=0.9)
# agg_opt = torch.optim.SGD(agg_model.parameters(), lr=0.001, momentum=0.9)
# %%
class FraudClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in shared_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(shared_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        shared_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        fed_train(shared_model, shared_opt, trainloader, epochs=5)
        return self.get_parameters(), num_examples['trainset'], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, f1_score = fed_test(shared_model, testloader)
        return float(loss), num_examples['testset'], {'f1_score': float(f1_score)}

fl.client.start_numpy_client('[::]:8080', client=FraudClient())

print('Client DONE!')