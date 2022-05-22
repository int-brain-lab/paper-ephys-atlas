# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from os.path import join
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
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate']
PID_EXCL = ['64d04585-67e7-4320-baad-8d4589fd18f7', '31d8dfb1-71fd-4c53-9229-7cd48bee07e4']

# Load in data
chan_volt = pd.read_parquet(args.data_path)
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
chan_volt = chan_volt.drop(PID_EXCL[0], level='pid')  # remove PIDs for testing
feature_arr = chan_volt[FEATURES].to_numpy()

# Initialize
clf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=20, max_leaf_nodes=1000,
                             n_jobs=-1)

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(chan_volt['acronym']), br.id[br.mappings['Allen']])
chan_volt['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Fit model
print('Fitting model..')
clf.fit(feature_arr, chan_volt['beryl_acronyms'].values)

# Save fitted model to disk
dump(clf, join(pathlib.Path(__file__).parent.resolve(), 'model.pkl'))
print('Fitted model saved to disk')