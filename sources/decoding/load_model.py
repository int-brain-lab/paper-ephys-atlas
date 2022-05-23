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
from joblib import dump, load
from model_functions import load_channel_data
import argparse
parser = argparse.ArgumentParser()

# Settings
PATH = 'C:\\Users\\guido\\Data\\EphysAtlas'
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth', 'theta', 'phi']
#PID = '64d04585-67e7-4320-baad-8d4589fd18f7'
PID = '31d8dfb1-71fd-4c53-9229-7cd48bee07e4'

# Load in data
data = load_channel_data(PATH)
data = data[data.index == PID]
feature_arr = data[FEATURES].to_numpy()

# Load in model
clf = load(join(PATH, 'model.pkl'))
region_predict = clf.predict(feature_arr)



