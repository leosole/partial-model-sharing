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
sys.path.append('.')
import config


df = pd.read_csv('data/lending_club_loan_two.csv')

df['TARGET'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})
df = df.drop('loan_status', axis=1)
df = df.drop('emp_title', axis=1)
df = df.drop('emp_length', axis=1)
df = df.drop('title', axis=1)
df = df.drop('grade', axis=1)
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc): #if we are missing the 'mort_acc' value
        return total_acc_avg[total_acc] #lookup the average value for the mort_acc based off the total_acc
    else:
        return mort_acc #if it's not missing, just return the mort_acc value

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1) #a function to two columns of Pandas dataframe
df = df.dropna()
df['term'] = df['term'].apply(lambda term: int(term[:3]))
dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1),dummies],axis=1)
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']],drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1),dummies],axis=1)
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1),dummies],axis=1)
df['zipcode'] = df['address'].apply(lambda adress : adress[-5:])
dummies = pd.get_dummies(df['zipcode'],drop_first=True)
df = pd.concat([df.drop('zipcode', axis=1),dummies],axis=1)
df = df.drop('address', axis=1)
df = df.drop('issue_d', axis=1)
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
#%%
y = df['TARGET']
df = df.drop('TARGET', axis=1)
X = pd.concat([df, y], axis=1)
X.to_csv('data/lending_club_loan_processed.csv')