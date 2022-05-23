# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from os.path import join
from model_functions import load_channel_data
from joblib import dump
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
import pathlib
import argparse
br = BrainRegions()
parser = argparse.ArgumentParser()

# Settings
parser.add_argument("-data_path", "--data_path", help="Path to training data")
args = parser.parse_args()
PID_EXCL = ['64d04585-67e7-4320-baad-8d4589fd18f7', '31d8dfb1-71fd-4c53-9229-7cd48bee07e4']
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth', 'theta', 'phi']

# Load in data
merged_df = load_channel_data(args.data_path)
feature_arr = merged_df[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=40, max_depth=20, max_leaf_nodes=15000,
                             n_jobs=-1, class_weight='balanced')

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(merged_df['acronym']), br.id[br.mappings['Allen']])
merged_df['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Fit model
print('Fitting model..')
clf.fit(feature_arr, merged_df['beryl_acronyms'].values)

# Save fitted model to disk
dump(clf, join(pathlib.Path(__file__).parent.resolve(), 'model.pkl'), compress=3)
print('Fitted model saved to disk')