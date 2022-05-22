# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pickle
from os.path import join
from sklearn.externals import joblib
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
br = BrainRegions()

# Settings
CLASSIFIER = 'forest'  # bayes or forest
N_FOLDS = 5
N_SHUFFLE = 100
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate']

# Load in data
chan_volt = pd.read_parquet('/home/guido/Data/ephys-atlas/channels_voltage_features.pqt')
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
feature_arr = chan_volt[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=100)
kfold = KFold(n_splits=N_FOLDS, shuffle=False)

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(chan_volt['acronym']), br.id[br.mappings['Allen']])
chan_volt['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Fit model
print('Fitting model..')
clf.fit(feature_arr, chan_volt['beryl_acronyms'].values)

# Save fitted model to disk
pickle.dump(clf, open(join('decoding', 'model_pickle.pkl'), 'wb'))
joblib.dump(clf, join('decoding', 'model_joblib.pkl'))
print('Fitted model saved to disk')