# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join, dirname, realpath, split
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from joblib import dump, load
from model_functions import load_channel_data, load_trained_model
import argparse
parser = argparse.ArgumentParser()

# Settings
ATLAS = 'cosmos'
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth']
#PID = '64d04585-67e7-4320-baad-8d4589fd18f7'
PID = '31d8dfb1-71fd-4c53-9229-7cd48bee07e4'

# Load in data
data = load_channel_data()
data = data[data.index == PID]
feature_arr = data[FEATURES].to_numpy()

# Load in model
clf = load_trained_model(atlas=ATLAS)
region_predict = clf.predict(feature_arr)

accuracy_score(data[f'{ATLAS}_acronyms'], region_predict)




