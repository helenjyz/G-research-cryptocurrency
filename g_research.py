#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:50:29 2022

@author: jiyuzhi
"""

#%%
import tensorflow
import json
import requests
import os
import re
from colorama import Fore, Back, Style
import plotly.express as px
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use('fivethirtyeight')
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#%%
asset_details = pd.read_csv('/Users/jiyuzhi/g-research-crypto-forecasting/asset_details.csv')
df_train = pd.read_csv('/Users/jiyuzhi/g-research-crypto-forecasting/train.csv')

#%%
#1439
df_train = df_train.set_index('timestamp')
df_train.index = pd.to_datetime(df_train.index, unit='s')
target_col = 'Target'
df_train= df_train.fillna(method='pad',axis=0)
btc = df_train[df_train["Asset_ID"]==1]
btc.drop(["Asset_ID"], axis = 'columns', inplace = True)
# btc = btc.reindex(range(btc.index[0],btc.index[-1]+60,60),method='pad')
# btc['timestamp'] = pd.to_datetime(btc['timestamp'], unit='s').apply(lambda x: x.strftime("%Y%m%d"))
# btc = btc.set_index("timestamp") 
btc_mini = btc.iloc[-20000:] 

#%%
from sklearn.model_selection import TimeSeriesSplit

#%%
n_splits = 5
tscv = TimeSeriesSplit(n_splits)


#%%
y = btc_mini['Target']
non_target_cols = list(set(btc_mini.columns) - set(['Target'] + ['timestamp']))
X = btc_mini[non_target_cols]

for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

train_data = btc_mini.iloc[train_index]
test_data = btc_mini.iloc[test_index]

#%%
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [btc]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);
    
#%%
line_plot(train_data[target_col], test_data[target_col], 'training', 'test', title='')




#%%
from xgboost.sklearn import XGBRegressor

#%%
from sklearn.metrics import mean_squared_error


#%%

model = XGBRegressor(verbosity=1)
model.fit(X_train,y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse)) # 0.001439 #0.000002071


#%%















