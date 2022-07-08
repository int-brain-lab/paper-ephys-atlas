# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from os.path import join, split
from model_functions import load_channel_data
from joblib import dump
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
import pathlib
import argparse
br = BrainRegions()
parser = argparse.ArgumentParser()

# Settings
parser.add_argument("-region", "--region", help="cosmos region to decode")
parser.add_argument("-atlas", "--atlas", help="beryl or allen")
parser.add_argument("-max_depth", "--max_depth", help="Max depth")
parser.add_argument("-n_trees", "--n_trees", help="Number of trees")
parser.add_argument("-max_leaf_nodes", "--max_leaf_nodes", help="Max leaf node")
args = parser.parse_args()
PID_EXCL = ['64d04585-67e7-4320-baad-8d4589fd18f7', '31d8dfb1-71fd-4c53-9229-7cd48bee07e4']
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth']

# Load in data
merged_df = load_channel_data()
merged_df = merged_df[merged_df['cosmos_acronyms'] == args.region]
merged_df = merged_df.drop(PID_EXCL, axis=0)  # drop PIDs used for evaluation
feature_arr = merged_df[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=int(args.n_trees),
                             max_depth=int(args.max_depth),
                             max_leaf_nodes=int(args.max_leaf_nodes),
                             n_jobs=-1, class_weight='balanced')

# Fit model
print('Fitting model..')
clf.fit(feature_arr, merged_df[f'{args.atlas}_acronyms'].values)

# Save fitted model to disk
dump(clf, join(split(pathlib.Path(__file__).parent.resolve())[0], 'trained_models',
               f'channels_model_{args.atlas}_{args.region}.pkl'), compress=3)
print('Fitted model saved to disk')