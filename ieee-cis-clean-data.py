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


data = pd.read_csv("data/train_transaction.csv")
data = data.sample(frac=1, random_state=config.random)
useful_features = list(data.iloc[:, 3:55].columns)

y = data.sort_values('TransactionDT')['isFraud']
X = data.sort_values('TransactionDT')[useful_features]
del data

categorical_features = [
    'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2',
    'P_emaildomain',
    'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

continuous_features = list(filter(lambda x: x not in categorical_features, X))
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

for f in tqdm(continuous_features):
    feature = X[f]
    log = lambda x: np.log10(x + 1 - min(0, x.min()))
    converter = ContinuousFeatureConverter(f, feature, log)
    feature_converters[f] = converter
    continuous_features_processed.append(converter.transform(feature))
    
continuous_train = pd.DataFrame({s.name: s for s in continuous_features_processed}).astype(np.float32)

continuous_train['isna_sum'] = continuous_train.isna().sum(axis=1)

continuous_train['isna_sum'] = (continuous_train['isna_sum'] - continuous_train['isna_sum'].mean())/continuous_train['isna_sum'].std()

isna_columns = []
for column in tqdm(continuous_features):
    isna = continuous_train[column].isna()
    if isna.mean() > 0.:
        continuous_train[column + '_isna'] = isna.astype(int)
        isna_columns.append(column)
        
continuous_train = continuous_train.fillna(0.)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def categorical_encode(df_train, categorical_features, n_values=50):
    df_train = df_train[categorical_features].astype(str)
    
    categories = []
    for column in tqdm(categorical_features):
        categories.append(list(df_train[column].value_counts().iloc[: n_values - 1].index) + ['Other'])
        values2use = categories[-1]
        df_train[column] = df_train[column].apply(lambda x: x if x in values2use else 'Other')
        
    
    ohe = OneHotEncoder(categories=categories)
    ohe.fit(df_train)
    df_train = pd.DataFrame(ohe.transform(df_train).toarray()).astype(np.float16)
    return df_train

train_categorical = categorical_encode(X, categorical_features) 
X = pd.concat([continuous_train, train_categorical, y], axis=1)
X.to_csv('data/train_transaction_processed.csv')