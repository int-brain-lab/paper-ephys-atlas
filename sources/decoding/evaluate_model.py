# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from iblutil.numerical import ismember
from model_functions import load_channel_data
from ibllib.atlas import BrainRegions
from joblib import dump
import argparse
br = BrainRegions()
parser = argparse.ArgumentParser()

# Settings
ATLAS = 'cosmos'

# Settings
N_FOLDS = 5
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth', 'theta', 'phi']

# Load in data
merged_df = load_channel_data()
feature_arr = merged_df[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=30, max_depth=25, max_leaf_nodes=10000,
                             n_jobs=-1, class_weight='balanced')
kfold = KFold(n_splits=N_FOLDS, shuffle=False)

# Decode brain regions
print('Decoding brain regions..')
feature_imp = np.empty((N_FOLDS, len(FEATURES)))
region_predict = np.empty(feature_arr.shape[0]).astype(object)
for i, (train_index, test_index) in zip(np.arange(N_FOLDS), kfold.split(feature_arr)):
    print(f'Fold {i+1} of {N_FOLDS}')
    clf.fit(feature_arr[train_index], merged_df[f'{ATLAS}_acronyms'].values[train_index])
    region_predict[test_index] = clf.predict(feature_arr[test_index])
    feature_imp[i, :] = clf.feature_importances_
acc = accuracy_score(merged_df[f'{ATLAS}_acronyms'].values, region_predict)
feature_imp = np.mean(feature_imp, axis=0)
print(f'Accuracy: {acc*100:.1f}%')

# Get accuracy per brain region
acc_region = pd.DataFrame(columns=['region', 'acc'])
for i, region in enumerate(merged_df[f'{args.atlas}_acronyms'].unique()):
    acc_region.loc[acc_region.shape[0]+1, 'region'] = region
    acc_region.loc[acc_region.shape[0], 'acc'] = accuracy_score(
        [region] * np.sum(merged_df[f'{args.atlas}_acronyms'] == region),
        region_predict[merged_df[f'{args.atlas}_acronyms'] == region])
acc_region = acc_region.sort_values('acc', ascending=False).reset_index(drop=True)

# %% Plot results
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.5), dpi=400)
ax1.bar(FEATURES, feature_imp)
ax1.set(ylabel='Feature importance')
ax1.set_xticklabels(FEATURES, rotation=90)

ax2.hist(acc_region['acc'] * 100, bins=30)
ax2.set(ylabel='Region count', xlabel='Accuracy (%)', xlim=[0, 20])

ax3.bar(acc_region[:10]['region'], acc_region[:10]['acc'] * 100)
ax3.set(ylabel='Accuracy (%)')
ax3.set_xticklabels(acc_region[:10]['region'], rotation=90)

plt.tight_layout()
sns.despine(trim=False)
plt.show()