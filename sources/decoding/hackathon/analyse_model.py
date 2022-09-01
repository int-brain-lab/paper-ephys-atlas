# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:05:48 2022

@author: Guido Meijer
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from ibllib.atlas import BrainRegions
from joblib import load
from model_functions import load_channel_data, load_trained_model
import matplotlib.pyplot as plt
import seaborn as sns

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
    test = chan_volt.loc[['31d8dfb1-71fd-4c53-9229-7cd48bee07e4', '64d04585-67e7-4320-baad-8d4589fd18f7'], : ]
else:
    test = chan_volt

feature_arr = test[FEATURES].to_numpy()
regions = test['cosmos_acronyms'].values

# Load model
clf = load_trained_model('channels', 'cosmos')

# Decode brain regions
print('Decoding brain regions..')
predictions = clf.predict(feature_arr)
probs = clf.predict_proba(feature_arr)

# histogram of response probabilities
certainties = probs.max(1)
plt.hist(certainties)
plt.close()

# plot of calibration, how certain are correct versus incorrect predicitions
plt.hist(certainties[regions == predictions], label='Correct predictions')
plt.hist(certainties[regions != predictions], label='Wrong predictions')
plt.title("Model calibration", size=24)
plt.legend(frameon=False, fontsize=16)
plt.ylabel("Occurences", size=21)
plt.xlabel("Prob for predicted region", size=21)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

sns.despine()
plt.tight_layout()
plt.savefig("/home/sebastian/Pictures/calibration")
plt.close()

# compute accuracy and balanced for our highly imbalanced dataset
acc = accuracy_score(regions, predictions)
bacc = balanced_accuracy_score(regions, predictions)
print(f'Accuracy: {acc*100:.1f}%')
print(f'Balanced accuracy: {bacc*100:.1f}%')


# compute confusion matrix
names = np.unique(np.append(regions, predictions))
cm = confusion_matrix(regions, predictions, labels=names)
cm = cm / cm.sum(1)[:, None]

cm_copy = cm.copy()

# list top n classifications
n = 10
np.max(cm[~np.isnan(cm)])
cm[np.isnan(cm)] = 0
for i in range(n):
    ind = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
    if ind[0] != ind[1]:
        print("Top {} classification, mistake: {} gets classified as {}".format(i+1, names[ind[0]], names[ind[1]]))
    else:
        print("Top {} classification, success: {} gets classified as {}".format(i+1, names[ind[0]], names[ind[1]]))
    cm[ind] = 0

# plot confusion matrix
plt.imshow(cm_copy)
plt.yticks(range(len(names)), names)
plt.xticks(range(len(names)), names, rotation='65')
plt.show()