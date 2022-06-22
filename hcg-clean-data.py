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


df = pd.read_csv("data/train_hcg.csv")
df = df.sample(frac=1, random_state=config.random)

#%%
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'Y':1,'N':0})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'Y':1,'N':0})
df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].replace({'Cash loans':1,'Revolving loans':0})


cat_vars = [col for col in df if df[col].dtype.name != 'float64' and df[col].dtype.name != 'float32' and df[col].dtype.name != 'int64' and len(df[col].unique()) < 150]
cat_sz = [(c, len(df[c].unique())+1) for c in cat_vars]
for cat in cat_vars:
    dummies = pd.get_dummies(df[cat], dummy_na=True)
    if np.nan in dummies.columns:
        dummies = dummies.drop([np.nan], axis=1)
    df = pd.concat([df,dummies], axis=1)
df = df.drop(cat_vars, axis=1)
y = df['TARGET']
df = df.drop(['TARGET'], axis=1)
df = pd.concat([df, y], axis=1)
#%%
df.to_csv('data/train_hcg_processed.csv')
# %%
