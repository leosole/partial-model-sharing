#%%
import collections
import numpy as np
import sys
import pandas as pd
import math
from math import sqrt
import datetime
from sklearn.model_selection import train_test_split

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import random
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import IPython.display as ipd
sys.path.append(".")
import config
#%%
data = pd.read_csv("data/train_portoseguro.csv")

data = data.drop(['id', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_reg_03', 'ps_car_14', 'ps_car_11_cat', 'ps_car_06_cat', 'ps_car_04_cat', 'ps_car_01_cat'], axis = 1)
data = data.replace(-1, np.nan)
data = data.replace('?', np.nan)

feature_bin = [f for f in data.columns if f.endswith('bin')] 
feature_cat = [f for f in data.columns if f.endswith('cat')] 
feature_els = [f for f in data.columns if (f not in feature_bin) & (f not in feature_cat) & (f not in ['id', 'target'])]

for f in (feature_bin + feature_cat):
    data[f].fillna(value=data[f].mode()[0], inplace=True)
    
for f in feature_els:
    data[f].fillna(value=data[f].mean(), inplace=True)

# drop remaining nan
data.dropna(inplace=True)

# managing categorical features
data = pd.get_dummies(columns=feature_cat, data=data)

cols = data.columns.to_list()
cols.append(cols.pop(0))
data = data[cols]

for c in data.columns:
  data = data[data[c] != -1]
#%%
data.to_csv('data/train_portoseguro_processed.csv')