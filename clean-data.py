#%%
import collections
import numpy as np
import sys
import pandas as pd
import math
from math import sqrt
import datetime
from sklearn.model_selection import train_test_split

import tensorflow_datasets as tfds
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import random
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import IPython.display as ipd
import matplotlib.pyplot as plt
sys.path.append(".")
from config_otto_tf import *
# path = 'data/ottoGroupProduct.csv'
# npath = 'data/ottoGroupProduct_proc.csv'
#%%
# ds = tfds.load('german_credit_numeric', split=['train'])
df = pd.read_csv(path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)


# %%
targets = [f'Class_{x}' for x in range(10)]
targets = targets[1:]
good_cols = {}
TH = 0.05
for target in targets: 
    corr = df.corrwith(df[target])
    cols = []
    for k, c in zip(corr.keys(),corr):
        if c > TH or c < -TH:
            cols.append(k)
    good_cols[target] = cols
all_good_cols = [x for x in cols for cols in good_cols]
good_set = set(all_good_cols)
good_set = [x for x in good_set if 'Class' not in x]
print(len(good_set))
ndf = df[good_set+targets]
# %%