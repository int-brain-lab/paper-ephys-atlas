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
from ibllib.atlas import BrainRegions
from joblib import dump
import argparse
br = BrainRegions()
parser = argparse.ArgumentParser()

# Settings
DATA_PATH = '/home/guido/Data/ephys-atlas/'
N_FOLDS = 5
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth', 'theta', 'phi']

# Load in data
chan_volt = pd.read_parquet(join(DATA_PATH, 'channels_voltage_features.pqt'))
chan_volt = chan_volt.drop(columns=['x', 'y', 'z'])
mm_coord = pd.read_parquet(join(DATA_PATH, 'coordinates.pqt'))
mm_coord.index.name = 'pid'
merged_df = pd.merge(chan_volt, mm_coord, how='left', on='pid')
merged_df = merged_df.loc[~merged_df['rms_ap'].isnull() & ~merged_df['x'].isnull()]  # remove NaNs
feature_arr = merged_df[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=40, max_depth=20, max_leaf_nodes=30000,
                             n_jobs=-1, class_weight='balanced')
kfold = KFold(n_splits=N_FOLDS, shuffle=False)

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(merged_df['acronym']), br.id[br.mappings['Allen']])
merged_df['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Decode brain regions
print('Decoding brain regions..')
feature_imp = np.empty((N_FOLDS, len(FEATURES)))
region_predict = np.empty(feature_arr.shape[0]).astype(object)
for i, (train_index, test_index) in zip(np.arange(N_FOLDS), kfold.split(feature_arr)):
    print(f'Fold {i+1} of {N_FOLDS}')
    clf.fit(feature_arr[train_index], merged_df['beryl_acronyms'].values[train_index])
    region_predict[test_index] = clf.predict(feature_arr[test_index])
    feature_imp[i, :] = clf.feature_importances_
acc = accuracy_score(merged_df['beryl_acronyms'].values, region_predict)
feature_imp = np.mean(feature_imp, axis=0)
print(f'Accuracy: {acc*100:.1f}%')

# Get accuracy per brain region
acc_region = pd.DataFrame(columns=['region', 'acc'])
for i, region in enumerate(chan_volt['beryl_acronyms'].unique()):
    acc_region.loc[acc_region.shape[0]+1, 'region'] = region
    acc_region.loc[acc_region.shape[0], 'acc'] = accuracy_score(
        [region] * np.sum(chan_volt['beryl_acronyms'] == region),
        region_predict[chan_volt['beryl_acronyms'] == region])
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