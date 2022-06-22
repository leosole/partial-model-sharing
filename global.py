#%%
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def train(model, opts, trainloader, epochs, device, criterion, metric_fn=ps.f1_score):
    history = {}
    history['loss'] = []
    history['metric'] = []
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
    return history

def test(model, testloader, criterion, device, metric_fn=ps.f1_score, print_test=True):
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

trainloader, testloader, num_examples, n_columns = ps.load_data(config.path, **config.server)

model = ps.Net([n_columns, *config.server_layers], config.dropout)
print('begin training')
opts = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
history = train(model, opts, trainloader, config.rounds, DEVICE, criterion)

print()
_, metric = test(model, testloader, criterion, DEVICE, print_test=True)
print('Client DONE!')
# if len(sys.argv) > 1:
#     with open(f'results/global_client-{sys.argv[1]}.txt', 'a') as f:
#         f.write(f'{metric}\n')
# else:
#     with open(f'results/global_client-.txt', 'a') as f:
#         f.write(f'{metric}\n')
# %%
