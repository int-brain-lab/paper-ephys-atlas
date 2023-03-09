#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:46:19 2023

@author: Joana Catarino, Kcénia Bougrova, Olivier Winter 
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
from one.remote import aws
from one.api import ONE

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 

from joblib import dump

from ibllib.atlas import BrainRegions


#%%   Benchmark datasets

'''

1) KS046_2020.12.03_P00         1a276285-8b0e-4cc9-9f0a-a3a002978724
2) SWC_043_2020.09.20_P00       1e104bf4-7a24-4624-a5b2-c2c8289c0de7
3) CSH_ZAD_029_2020.09.17_PO1   5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e
4) NYU-40_2021.04.15_P01        5f7766ce-8e2e-410c-9195-6bf089fea4fd
5) SWC_043_2020.09.21_P01       6638cfb3-3831-4fc2-9327-194b76cf22e1
6) SWC_038_2020.07.30_P01       749cb2b7-e57e-4453-a794-f6230e4d0226
7) KS096_2022.06.17_P00         d7ec0892-0a6c-4f4f-9d8f-72083692af5c
8) SWC_043_2020.09.21_P00       da8dfec1-d265-44e8-84ce-6ae9c109b8bd
9) DY_016_2020.09.12_P00        dab512bd-a02d-4c1f-8dbc-9155a163efc0
10) ZM_2241_2020.01.27_P00      dc7e9403-19f7-409f-9240-05ee57cb7aea
11) ZM_1898_2019.12.10_P00      e8f9fba4-d151-4b00-bee7-447f0f3e752c
12) UCLA033_2022.02.15_P01      eebcaf65-7fa4-4118-869d-a084e84530e2
13) CSH_ZAD_025_2020.08.03_P01  fe380793-8035-414e-b000-09bfe5ece92a

'''
#%%
# Get the different dataframes with all the features


LOCAL_DATA_PATH = Path("/home/joana/Desktop/IBL/Ephys_Atlas/dataframes_features")

# The AWS private credentials are stored in Alyx, so that only one authentication is required
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='online')
s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
aws.s3_download_folder("aggregates/bwm",
                       LOCAL_DATA_PATH,
                       s3=s3, bucket_name=bucket_name)

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('raw_ephys_features.pqt'))


plt.figure()
sns.countplot(data=df_probes, x='histology', palette='deep')

plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')

#%%

regions = BrainRegions()
LOCAL_DATA_PATH = Path("/home/joana/Desktop/IBL/Ephys_Atlas/dataframes_features")
OUT_PATH = Path("/home/joana/Desktop/IBL/Ephys_Atlas/ephys_atlas_features")

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('raw_ephys_features.pqt'))

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
aids_cosmos = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
aids_beryl = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')
df_voltage['beryl_id'] = aids_beryl
df_voltage['cosmos_id'] = aids_cosmos


# Decode brain regions
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'spike_rate', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)

df_voltage = df_voltage.reset_index()

#%%

# Training the model

br = BrainRegions()
acronyms = br.id2acronym(aids_cosmos) #This changes the id (number of a region) to the acronym name 

n_trees = 30
max_depth = 25
max_leaf_nodes = 10000  # to be optimized! change the values until find when the model loses performance


Benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724', 
                  '1e104bf4-7a24-4624-a5b2-c2c8289c0de7', 
                  '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e', 
                  '5f7766ce-8e2e-410c-9195-6bf089fea4fd', 
                  '6638cfb3-3831-4fc2-9327-194b76cf22e1', 
                  '749cb2b7-e57e-4453-a794-f6230e4d0226', 
                  'd7ec0892-0a6c-4f4f-9d8f-72083692af5c', 
                  'da8dfec1-d265-44e8-84ce-6ae9c109b8bd', 
                  'dab512bd-a02d-4c1f-8dbc-9155a163efc0', 
                  'dc7e9403-19f7-409f-9240-05ee57cb7aea', 
                  'e8f9fba4-d151-4b00-bee7-447f0f3e752c', 
                  'eebcaf65-7fa4-4118-869d-a084e84530e2', 
                  'fe380793-8035-414e-b000-09bfe5ece92a']

df_voltage2 = df_voltage
for i in Benchmark_pids: 
     df_voltage2 = df_voltage2[df_voltage2.pid != i] # != excludes the pids you don't want to include in your dataframe
     

df_voltage2 = df_voltage2.reset_index(drop=True)


aids_cosmos = regions.remap(df_voltage2['atlas_id'], source_map='Allen', target_map='Cosmos')
aids_beryl = regions.remap(df_voltage2['atlas_id'], source_map='Allen', target_map='Beryl')
df_voltage2['beryl_id'] = aids_beryl
df_voltage2['cosmos_id'] = aids_cosmos


X_train = df_voltage2.loc[:, x_list].values
y_train = acronyms # we are training the model to already give the output as an acronym instead of an id

# Model

classifier=RandomForestClassifier(random_state=42, verbose=True, n_estimators=n_trees,
                                  max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

clf = classifier.fit(X_train, y_train)


#%%

# Testing the model

df_voltage3 = df_voltage[df_voltage.pid == '749cb2b7-e57e-4453-a794-f6230e4d0226']  #Change this PID depending on the one we want to test
df_voltage3 = df_voltage3.reset_index(drop=True) 

X_test = df_voltage3.loc[:, x_list].values

predict_region = clf.predict(X_test)

# Modify id to acronyms and score the model for the selected tested pid

aids_cosmos2 = regions.remap(df_voltage3['atlas_id'], source_map='Allen', target_map='Cosmos')
aids_beryl2 = regions.remap(df_voltage3['atlas_id'], source_map='Allen', target_map='Beryl')
acronyms2 = br.id2acronym(aids_cosmos2)


ids_pred = br.acronym2id(predict_region) # This changes the id  (number of a region) to the acronym name 


score_model = clf.score(ids_pred, aids_cosmos2)

#%%  

#  Plots

# Compute confusion matrix 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

vmax=1  # max value 
vmin= 0 # min value

a = regions.id2acronym(aids_cosmos2,mapping='Cosmos')
b = regions.id2acronym(ids_pred,mapping='Cosmos')
#c = regions.id2acronym(aids_beryl2,mapping='Beryl')
#d = regions.id2acronym(ids_pred,mapping='Beryl')

names = np.unique(np.append(a, b)) # labels for cosmos
#names = np.unique(np.append(c, d)) # labels for beryl
#cm = confusion_matrix(aids_beryl2, ids_pred) #Beryl atlas
cm = confusion_matrix(aids_cosmos2, ids_pred) #Cosmos atlas
cmfinal = cm/100

f, (ax1) = plt.subplots(1, 1, dpi=500, figsize=(9,4))
ax1.imshow(cmfinal,cmap="viridis")
sns.heatmap(cmfinal,xticklabels=names,yticklabels=names, vmax=vmax, vmin=vmin)
ax1.set_title('SWC_038_2020.07.30_P01', fontsize=12, pad=12) 
ax1.set_ylabel('Predicted', rotation=90, labelpad=12)
ax1.set_xlabel('Real', rotation=0, labelpad=12)
sns.despine(trim=False)
plt.grid(color='gray', linewidth=0.5)
ax1.set_yticks(range(len(names)), names)
ax1.set_xticks(range(len(names)), names)

plt.tight_layout()

plt.savefig('/home/joana/Desktop/Ephys atlas/Figures Confusion Matrix/SWC_038_2020.07.30_P01_Cosmos_ConfusionMatrix.png')
plt.savefig('/home/joana/Desktop/Ephys atlas/Figures Confusion Matrix/SWC_038_2020.07.30_P01_Cosmos_ConfusionMatrix.pdf')


 # Plot similar to alignment gui
 
 
from brainbox.ephys_plots import plot_brain_regions
import matplotlib.pyplot as plt

df_channels = df_channels.reset_index(drop=False)
df_channels = df_channels[df_channels.pid == '749cb2b7-e57e-4453-a794-f6230e4d0226']
df_channels = df_channels.reset_index(drop=True)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 16), dpi=500)

plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=ax1, title='Real')
ax1.tick_params(labelsize=18)
ax1.set_title('Real',fontsize=18)

plot_brain_regions(ids_pred, channel_depths=df_voltage3['axial_um'].values,
                           brain_regions=br, display=True, ax=ax2)
ax2.tick_params(labelsize=18)
ax2.set_title('Predicted',fontsize=18)
  
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)


plt.savefig('/home/joana/Desktop/Ephys atlas/Figures probe alignment/SWC_038_2020.07.30_P01_Cosmos_ProbeAlignment.png')
plt.savefig('/home/joana/Desktop/Ephys atlas/Figures probe alignment/SWC_038_2020.07.30_P01_Cosmos_ProbeAlignment.pdf')


'''
To plot only one:

f, (ax1) = plt.subplots(1, 1, figsize=(4, 16))

plot_brain_regions(df_channels['atlas_id'].values, channel_depths=df_channels['axial_um'].values,
                           brain_regions=br, display=True, ax=ax1, title='self.histology')

'''

