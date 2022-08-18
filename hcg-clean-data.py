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
# https://www.kaggle.com/competitions/home-credit-default-risk/overview
#%%
df = pd.read_csv("data/train_hcg.csv")
df = df.sample(frac=1, random_state=config.random)
# df = df.dropna()
#%%
df = df[df['CODE_GENDER'] != 'XNA']
df = df[df['AMT_INCOME_TOTAL'] < 20000000] 
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) 
df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) 
def get_age_label(days_birth):
        """ Return the age group label (int). """
        age_years = -days_birth / 365
        if age_years < 27: return 1
        elif age_years < 40: return 2
        elif age_years < 50: return 3
        elif age_years < 65: return 4
        elif age_years < 99: return 5
        else: return 0
    # Categorical age - based on target=1 plot
df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'Y':1,'N':0})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'Y':1,'N':0})
df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].replace({'Cash loans':1,'Revolving loans':0})

df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

# Credit ratios
df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

# Income ratios
df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']

# Time ratios
df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
df = df.drop(['DAYS_BIRTH'], axis=1)
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
df = df.dropna()
df.to_csv('data/train_hcg_processed.csv')
# %%
