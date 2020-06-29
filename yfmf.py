import os
from os import walk
from os import path
import tensorflow as tf
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import scipy as scp
from math import erf
import yfinance as yf
import collections
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
from datetime import date
from itertools import cycle, islice
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
import mplfinance as mpf
import copy


def plot_stocks(data,lim1=-2,lim2=-1,means=(1),name="stock",frmaesize=(10,6),show_nonetrading=True,show_volume=True):
    daily=data[lim1:lim2]
    daily.index.name = 'Date'
    daily.shape
    mpf.plot(daily,type='candle',mav=means,volume=show_volume,show_nontrading=show_nonetrading,figsize=frmaesize,title=name)   

def plot_features(dfs_scaled,dfs_cp):
    
    for key in dfs_scaled.keys():
        fig, ax = plt.subplots(1,2, figsize = (5,3))        
        ax[0].plot(dfs_scaled[key],'*', label = key + '-scaled')
        ax[1].plot(dfs_cp[key],'*', label = key+'-org')
        ax[0].legend()
        ax[1].legend()

def split_data(df,targ,p_train,p_dev):
    dataset = tf.data.Dataset.from_tensor_slices((df.values, targ.values))
    for feat, targ in dataset.take(5):
        print ('Features: {}, Target: {}'.format(feat, targ))
    len_set=len(df)
    len_train=int(np.floor(len_set*p_train))
    len_dev=int(np.floor(len_set*p_dev))
    len_test=int(len_set-len_dev-len_train)
    train= dataset.take(len_train)
    rem1= dataset.skip(len_train)
    dev= rem1.take(len_dev)
    rem2 = rem1.skip(len_dev)
    test = rem2.take(-1)
    print(len_set,len_train,len_dev,len_test)
    return train,dev,test,len_train,len_dev,len_test

def get_compiled_model(input_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_size, activation='relu'),
    tf.keras.layers.Dense(input_size, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model