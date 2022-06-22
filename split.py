#%%
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flwr as fl

sys.path.append(".")
from importlib import import_module
# if len(sys.argv) > 1:
#     config = import_module(sys.argv[1])
# else: 
    # import config
config = import_module('config_ieee2')
import partial_split as ps
#%%
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

history = {}
history['loss'] = []
history['metric'] = []

def train(model, opts, trainloader, epochs, criterion, device=DEVICE, metric_fn=ps.f1_score):
    model.train()
    for i in range(epochs):
        num_right = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opts.zero_grad()
            outputs = model(x)
            # outputs = outputs.view(-1)
            loss = criterion(outputs, y.long())
            loss.backward()
            history['loss'].append(loss.item())
            opts.step()
            _, labels = torch.max(outputs, 1)
            num_right += np.sum(labels.data.numpy() == y.long().data.numpy())
            # preds = np.round_(outputs.detach().numpy())
            # actuals.extend(y)
            # predictions.extend(preds)
        metric = num_right / len(trainloader.dataset) 
        history['metric'].append(metric)
        print(f'\rTRAIN [{i/epochs*100:.1f}%] Metric: {metric:.4f} \t Loss: {loss:.4f}', end='')

def test(model, testloader, criterion, device=DEVICE, metric_fn=ps.f1_score, print_test=False):
    loss = 0.0
    num_right = 0
    model.eval()
    with torch.no_grad():
        actuals = []
        predictions = []
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            # outputs = outputs.view(-1)
            loss += criterion(outputs, y.long()).item()
            _, labels = torch.max(outputs, 1)
            num_right += np.sum(labels.data.numpy() == y.long().data.numpy())

    metric = num_right / len(testloader.dataset)
    if print_test:
        print(f'TEST | Metric: {metric:.4f} \t Loss: {loss:.4f}')
    return loss, metric

shared_columns = [col for col in config.client1['columns'] if col in config.client2['columns']]

if sys.argv[1] == '1':
    trainloader, testloader, num_examples, _ = ps.load_data(config.path, **config.client1)
    ind_columns = [col for col in config.client1['columns'] if col not in shared_columns]

if sys.argv[1] == '2':
    trainloader, testloader, num_examples, _ = ps.load_data(config.path, **config.client2)
    ind_columns = [col for col in config.client2['columns'] if col not in shared_columns]

shared_model = ps.Net([len(shared_columns), *config.shared_layers], config.dropout)
ind_model = ps.Net([len(ind_columns), *config.ind_layers], config.dropout)
agg_model = ps.Net([shared_model.last + ind_model.last, *config.agg_layers], config.dropout)
splitNN = ps.SplitNN({'ind_model': ind_model, 'shared_model': shared_model, 'agg_model': agg_model}, ind_columns, shared_columns)
print('begin training')
opts = torch.optim.Adam(splitNN.parameters(), lr=config.lr, weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss()
class FraudClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in splitNN.shared_model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(splitNN.shared_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        splitNN.shared_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, configuration):
        self.set_parameters(parameters)
        train(splitNN, opts, trainloader, config.epochs, criterion)
        return self.get_parameters(), num_examples['trainset'], {}

    def evaluate(self, parameters, configuration):
        self.set_parameters(parameters)
        loss, accuracy = test(splitNN, testloader, criterion)
        return float(loss), num_examples['testset'], {'accuracy': float(accuracy)}
# criterion = nn.BCELoss()
fl.client.start_numpy_client('[::]:8080', client=FraudClient())
print('Client DONE!')
_, metric = test(splitNN, testloader, criterion, print_test=True)
if len(sys.argv) > 1:
    with open(f'results/split_client-{sys.argv[1]}.txt', 'a') as f:
        f.write(f'{metric}\n')
else:
    with open(f'results/split_client-.txt', 'a') as f:
        f.write(f'{metric}\n')
# %%
