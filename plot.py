#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from glob import glob
#%%
data = {}
dataset = 'ieee'
shared = 200
total_col = 324
sim_num = 7
if shared:
    for file in glob(f'results/tf*{dataset}*{shared}*.txt'):
        with open(file, 'r') as f:
            lines = []
            i = 0
            for line in f:
                if i < sim_num:
                    lines.append(float(line.split('\n')[0]))
                i += 1
            label = file.split('/')[-1].split('-')[0].replace('tf_', '').replace('_', ' ')
            data[label] = lines
    for file in glob(f'results/tf_global*{dataset}.txt'):
        with open(file, 'r') as f:
            lines = []
            i = 0
            for line in f:
                if i < sim_num:
                    lines.append(float(line.split('\n')[0]))
                i += 1
            label = file.split('/')[-1].split('-')[0].replace('tf_', '').replace('_', ' ')
            data[label] = lines
else:
    for file in glob(f'results/{dataset}/*.txt'):
        with open(file, 'r') as f:
            lines = []
            i = 0
            for line in f:
                if i < sim_num:
                    lines.append(float(line.split('\n')[0]))
                i += 1
            label = file.split('/')[-1].split('-')[0].replace('tf_', '').replace('_', ' ')
            data[label] = lines
            shared = file.split('_')[-1].split('.')[0]
df = pd.DataFrame.from_dict(data)
# %%
upper_bound = df.mean().max()
def normalize(x):
    return x/upper_bound
def denormalize(x):
    return x*upper_bound
df2 = df/upper_bound
sns.set_theme(style="darkgrid")
sns.set_context('poster')
fig, ax1 = plt.subplots()
fig.set_size_inches(16, 9)
order = ['local 1', 'local 2', 'transfer 1', 'transfer 2', 'partial 1', 'partial 2', 'global']
df = df[order]
sns.pointplot(data=df, join=False, palette="Paired", ci='sd', ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_ylabel('F1 score')
ax2 = ax1.secondary_yaxis('right', functions=(normalize, denormalize))
ax2.set_ylabel('Normalized F1 score')
ax1.set_title(f'{dataset.upper()} Dataset ({100*shared/total_col:.0f}% shared columns)')
labels = []
for label, mean in zip(df.columns, df.mean()):
    labels.append(label + f':  {mean:.3f}')
    print(mean)
plt.legend(labels=labels, edgecolor=(1,1,1,0), borderpad=0.8)
# %%
