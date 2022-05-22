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
from joblib import dump, load
import argparse
br = BrainRegions()
parser = argparse.ArgumentParser()

# Settings
parser.add_argument("-data_path", "--data_path", help="Path to training data")
parser.add_argument("-model_path", "--model_path", help="Path to model")

args = parser.parse_args()
N_FOLDS = 10
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate']

# Load in data
chan_volt = pd.read_parquet(args.data_path)
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
test = chan_volt.xs('071f02e7-752a-4094-af79-8dd764e9d85d', level='pid')
feature_arr = test[FEATURES].to_numpy()

# Load model
clf = load(args.model_path)

# Remap to Beryl atlas
_, inds = ismember(br.acronym2id(chan_volt['acronym']), br.id[br.mappings['Allen']])
chan_volt['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']

# Decode brain regions
print('Decoding brain regions..')

acc = accuracy_score(test['beryl_acronyms'].values,
                     clf.predict(feature_arr))
print(f'Accuracy: {acc*100:.1f}%')
print(f'Chance level: {(1/chan_volt["beryl_acronyms"].unique().shape[0])*100:.1f}%')
