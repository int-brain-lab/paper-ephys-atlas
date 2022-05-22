# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import pathlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
from joblib import dump
import argparse
br = BrainRegions()
parser = argparse.ArgumentParser()

# Settings
parser.add_argument("-data_path", "--data_path", help="Path to training data")
parser.add_argument("-n_folds", "--n_folds", help="Number of folds")
args = parser.parse_args()
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate']

# Load in data
chan_volt = pd.read_parquet(args.data_path)
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
feature_arr = chan_volt[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=20, max_leaf_nodes=1000,
                             n_jobs=-1)
kfold = KFold(n_splits=args.n_folds, shuffle=False)

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(chan_volt['acronym']), br.id[br.mappings['Allen']])
chan_volt['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Decode brain regions
print('Decoding brain regions..')
feature_imp = np.empty((args.n_folds, len(FEATURES)))
region_predict = np.empty(feature_arr.shape[0])
for i, (train_index, test_index) in zip(np.arange(args.n_folds), kfold.split(feature_arr)):
    clf.fit(feature_arr[train_index], chan_volt['beryl_acronyms'].values[train_index])
    region_predict[test_index] = clf.predict(feature_arr[test_index])
    feature_imp[i, :] = clf.feature_importances_
acc = accuracy_score(chan_volt['beryl_acronyms'].values, region_predict)
feature_imp = np.mean(feature_imp, axis=0)
print(f'Accuracy: {acc*100:.1f}%')

# Get accuracy per brain region
acc_region = dict()
for i, region in enumerate(chan_volt['beryl_acronyms'].unique()):
    acc_region[region] = accuracy_score([region] * np.sum(chan_volt['beryl_acronyms'] == region),
                                        region_predict[chan_volt['beryl_acronyms'] == region])
acc_region = dict(sorted(acc_region.items(), key=lambda item: item[1]))
    
# Plot results
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 1.75), dpi=500)
ax1.bar(FEATURES, feature_imp)
ax1.set(ylabel='Feature importance')
ax1.set_xticklabels(FEATURES, rotation=45)

#ax2.
