# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import pathlib
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
parser.add_argument("-classifier", "--classifier", help="forest or bayes")
parser.add_argument("-max_depth", "--max_depth", help="Max depth")
parser.add_argument("-n_trees", "--n_trees", help="")
parser.add_argument("-max_leaf_nodes", "--max_leaf_nodes", help="")
args = parser.parse_args()
classifier = args.classifier
CLASSIFIER = 'forest'  # bayes or forest
N_FOLDS = 10
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate']

# Load in data
chan_volt = pd.read_parquet(args.data_path)
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
feature_arr = chan_volt[FEATURES].to_numpy()

# Initialize
if args.classifier == 'forest':
    clf = RandomForestClassifier(random_state=42, n_estimators=int(args.n_trees),
                                 max_depth=int(args.max_depth),
                                 max_leaf_nodes=int(args.max_leaf_nodes),
                                 n_jobs=-1)
elif args.classifier == 'bayes':
    clf = GaussianNB()
kfold = KFold(n_splits=N_FOLDS, shuffle=False)

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(chan_volt['acronym']), br.id[br.mappings['Allen']])
chan_volt['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Decode brain regions
print('Decoding brain regions..')
feature_imp = np.empty((N_FOLDS, len(FEATURES)))
train_index, test_index = next(kfold.split(feature_arr))
clf.fit(feature_arr[train_index], chan_volt['beryl_acronyms'].values[train_index])
acc = accuracy_score(chan_volt['beryl_acronyms'].values[test_index],
                     clf.predict(feature_arr[test_index]))
print(f'Accuracy: {acc*100:.1f}%')
print(f'Chance level: {(1/chan_volt["beryl_acronyms"].unique().shape[0])*100:.1f}%')

# Save fitted model to disk
dump(clf, join(pathlib.Path(__file__).parent.resolve(), 'test_model.pkl'))
print('Fitted model saved to disk')