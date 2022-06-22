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

data = pd.read_csv("data/train_santander_processed.csv")
data = data.sample(frac=1, random_state=config.random)
data.head()
#%%