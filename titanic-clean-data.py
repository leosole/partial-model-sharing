#%%
import collections
import numpy as np
import sys
import pandas as pd

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
from sklearn.preprocessing import StandardScaler
import IPython.display as ipd
sys.path.append(".")
import config


df = pd.read_csv("data/train_titanic.csv")
dmean = df["Age"].dropna().mean()
df["Age"] = df["Age"].fillna(dmean)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df["Sex"])
df["Sex"] = le.transform(df["Sex"])

df_title = [i.split(',')[1].split('.')[0].strip() for i in df['Name']]
# df['Title'] = pd.Series(df_title)
# df['Title'].value_counts()
# df['Title'] = df['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], 'Rare')
df['FamilyS'] = df['SibSp'] + df['Parch'] + 1
def family(x):
    if x < 2:
        return 'Single'
    elif x == 2:
        return 'Couple'
    elif x <= 4:
        return 'InterM'
    else:
        return 'Large'
    
df['FamilyS'] = df['FamilyS'].apply(family)
df['Embarked'].fillna('S',inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
# for f in ['Embarked', 'Title', 'FamilyS']:
# for f in ['Embarked', 'FamilyS']:    
#     dummies = pd.get_dummies(df[f], dummy_na=True)
#     if np.nan in dummies.columns:
#         dummies = dummies.drop([np.nan], axis=1)
#     df = pd.concat([df,dummies], axis=1)
y = df['Survived']
# df = df.drop(['PassengerId', 'Survived', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket','Embarked', 'Title', 'FamilyS'], axis=1)
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']]
df = pd.concat([df, y], axis=1)
# X = pd.concat([continuous_train, train_categorical, y], axis=1)
df.to_csv('data/train_titanic_processed.csv')