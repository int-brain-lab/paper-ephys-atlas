# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ibllib.atlas import BrainRegions
from joblib import load
from model_functions import load_channel_data, load_trained_model
import matplotlib.pyplot as plt

br = BrainRegions()

# Settings
FEATURES = ['psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf',
            'spike_rate', 'axial_um', 'x', 'y', 'depth']

# Load in data
chan_volt = load_channel_data()
# chan_volt = pd.read_parquet("/home/sebastian/Downloads/FlatIron/tables/channels_voltage_features.pqt")
chan_volt = chan_volt.loc[~chan_volt['rms_ap'].isnull()]  # remove NaNs
# 31d8dfb1-71fd-4c53-9229-7cd48bee07e4 64d04585-67e7-4320-baad-8d4589fd18f7
if True:
    test = chan_volt.loc['31d8dfb1-71fd-4c53-9229-7cd48bee07e4', :]
else:
    test = chan_volt

feature_arr = test[FEATURES].to_numpy()
regions = test['beryl_acronyms'].values

# Load model
clf = load_trained_model('channels', 'cosmos')

# Decode brain regions
print('Decoding brain regions..')
predictions = clf.predict(feature_arr)
probs = clf.predict_proba(feature_arr)

depths = feature_arr[:, FEATURES.index('axial_um')]

region_to_num = {}
region_counter = 0
assert np.array_equal(np.repeat(np.arange(192), 2) + 1, depths / 20), "Depths are not in required format"
probe_regions = np.zeros((192, 2))  # is this always correct?
probe_predictions = np.zeros((192, 2))
for i in range(depths.shape[0]):
    if i < 20:
        print(i // 2, int(i % 2))
    if predictions[i] not in region_to_num:
        region_to_num[predictions[i]] = region_counter
        region_counter += 1
    if regions[i] not in region_to_num:
        region_to_num[regions[i]] = region_counter
        region_counter += 1

    probe_regions[i // 2, int(i % 2)] = region_to_num[regions[i]]
    probe_predictions[i // 2, int(i % 2)] = region_to_num[predictions[i]]

plt.subplot(1, 2, 1)
plt.imshow(probe_regions)


plt.subplot(1, 2, 2)
plt.imshow(probe_predictions)

plt.show()