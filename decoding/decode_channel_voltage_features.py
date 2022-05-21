# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Settings
N_FOLDS = 5
N_SHUFFLE = 100
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate']

# Load in data
chan_volt = pd.read_parquet('F://channels_voltage_features.pqt')
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
feature_arr = chan_volt[FEATURES].to_numpy()

# Initialize 
rf = RandomForestClassifier(random_state=42)
kfold = KFold(n_splits=N_FOLDS, shuffle=False)

# Decode brain regions
region_predict = np.empty(chan_volt.shape[0]).astype(object)
for train_index, test_index in kfold.split(feature_arr):
    rf.fit(feature_arr[train_index], chan_volt['acronym'].values[train_index])
    region_predict[test_index] = rf.predict(feature_arr[test_index])
acc = accuracy_score(chan_volt['acronym'].values, region_predict)
 
# Decode lab with shuffled lab labels
shuf_acc = np.empty(N_SHUFFLE)
for i in range(N_SHUFFLE):
    region_shuf = shuffle(chan_volt['acronym'].values)
    region_predict_shuffle = np.empty(chan_volt.shape[0]).astype(object)
    for train_index, test_index in kfold.split(feature_arr):
        rf.fit(feature_arr[train_index], region_shuf[train_index])
        region_predict_shuffle[test_index] = rf.predict(feature_arr[test_index])
    shuf_acc[i] = accuracy_score(chan_volt['lab'].values, region_predict_shuffle)