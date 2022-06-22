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
df = pd.read_csv("data/train_portoseguro.csv")
for c in df.columns:
  df = df[df[c] != -1]
df = df.reset_index()
y = df.sort_values('id')['target']
X = df.sort_values('id').drop(['target', 'id'], axis=1)
#%%
cols = X.columns.to_list()
categorical = [x for x in cols if 'cat' in x]
ordinal = [x for x in cols if x not in categorical]
# %%
class ContinuousFeatureConverter:
    def __init__(self, name, feature, log_transform):
        self.name = name
        self.skew = feature.skew()
        self.log_transform = log_transform
        
    def transform(self, feature):
        if self.skew > 1:
            feature = self.log_transform(feature)
        
        mean = feature.mean()
        std = feature.std()
        return (feature - mean)/(std + 1e-6)

from tqdm.autonotebook import tqdm

feature_converters = {}
continuous_features_processed = []

for f in tqdm(ordinal):
    feature = X[f]
    log = lambda x: np.log10(x + 1 - min(0, x.min()))
    converter = ContinuousFeatureConverter(f, feature, log)
    feature_converters[f] = converter
    continuous_features_processed.append(converter.transform(feature))
    
continuous_train = pd.DataFrame({s.name: s for s in continuous_features_processed}).astype(np.float32)

continuous_train['isna_sum'] = continuous_train.isna().sum(axis=1)

continuous_train['isna_sum'] = (continuous_train['isna_sum'] - continuous_train['isna_sum'].mean())/continuous_train['isna_sum'].std()

isna_columns = []
for column in tqdm(ordinal):
    isna = continuous_train[column].isna()
    if isna.mean() > 0.:
        continuous_train[column + '_isna'] = isna.astype(int)
        isna_columns.append(column)
        
continuous_train = continuous_train.fillna(0.)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def categorical_encode(df_train, categorical, n_values=50):
    df_train = df_train[categorical].astype(str)
    
    categories = []
    for column in tqdm(categorical):
        categories.append(list(df_train[column].value_counts().iloc[: n_values - 1].index) + ['Other'])
        values2use = categories[-1]
        df_train[column] = df_train[column].apply(lambda x: x if x in values2use else 'Other')
        
    
    ohe = OneHotEncoder(categories=categories)
    ohe.fit(df_train)
    df_train = pd.DataFrame(ohe.transform(df_train).toarray()).astype(np.float16)
    return df_train

train_categorical = categorical_encode(X, categorical) 
X = pd.concat([continuous_train, train_categorical, y], axis=1)
#%%
X = X.drop(['index'], axis=1)
X.to_csv('data/train_portoseguro_processed.csv')
X
# %%
