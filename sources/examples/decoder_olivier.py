#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:46:19 2023

@author: Joana Catarino, Kcénia Bougrova, Olivier Winter

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

"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from iblutil.numerical import ismember

from ephys_atlas.interactives import plot_probas
from brainbox.ephys_plots import plot_brain_regions
from ibllib.atlas import BrainRegions

regions = BrainRegions()
LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/features_tables")
OUT_PATH = Path("/datadisk/Data/paper-ephys-atlas/decoding")

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('raw_ephys_features.pqt'))

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
# remapping the lowest level of channel location to higher levels mappings
df_voltage['cosmos_id'] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
df_voltage['beryl_id'] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')


# selection and scaling of features
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'spike_rate', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)

df_voltage = df_voltage.reset_index()

## %%

# Training the model
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

df_voltage2 = df_voltage.copy()
for i in Benchmark_pids:
     df_voltage2 = df_voltage2[df_voltage2.pid != i]  # != excludes the pids you don't want to include in your dataframe

a, _ = ismember(df_voltage2['cosmos_id'], regions.acronym2id(['void', 'root']))
df_voltage2 = df_voltage2.loc[~a]
df_voltage2 = df_voltage2.reset_index(drop=True)

X_train = df_voltage2.loc[:, x_list].values
y_train = df_voltage2['acronym'].values  # we are training the model to already give the output as an acronym instead of an id

# Model

classifier = RandomForestClassifier(random_state=42, verbose=True, n_estimators=n_trees,
                                    max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

clf = classifier.fit(X_train, y_train)


## %%
# Testing the model on
for pid in Benchmark_pids:
    # pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
    df_pid = df_voltage[df_voltage.pid == pid]
    df_pid = df_pid.reset_index(drop=True)
    nc = df_pid.shape[0]
    X_test = df_pid.loc[:, x_list].values

    # we output both the most likely region and all of the probabilities for each region
    predict_region = clf.predict(X_test)
    predict_regions_proba = pd.DataFrame(clf.predict_proba(X_test).T)

    # this is the probability of the most likely region
    max_proba = np.max(predict_regions_proba.values, axis=0)

    # TODO aggregate per depth, not per channel !
    # we remap and aggregate probabilites
    predict_regions_proba['beryl_aids'] = regions.remap(regions.acronym2id(clf.classes_), source_map='Allen', target_map='Beryl')
    predict_regions_proba['cosmos_aids'] = regions.remap(regions.acronym2id(clf.classes_), source_map='Allen', target_map='Cosmos')
    probas_beryl = predict_regions_proba.groupby('beryl_aids').sum().drop(columns='cosmos_aids').T
    probas_cosmos = predict_regions_proba.groupby('cosmos_aids').sum().drop(columns='beryl_aids').T


    predictions_remap_cosmos = regions.remap(regions.acronym2id(predict_region), source_map='Allen', target_map='Cosmos')
    predictions_remap_beryl = regions.remap(regions.acronym2id(predict_region), source_map='Allen', target_map='Beryl')

    sns.set_theme()
    f, axs = plt.subplots(1, 6, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 4, 1, 1, 4]})

    plot_brain_regions(df_pid['cosmos_id'], channel_depths=df_pid['axial_um'].values,
                               brain_regions=regions, display=True, ax=axs[0], title='Aligned')
    plot_brain_regions(predictions_remap_cosmos, channel_depths=df_pid['axial_um'].values,
                               brain_regions=regions, display=True, ax=axs[1], title='Classified')
    plot_probas(probas_cosmos, legend=False, ax=axs[2])

    plot_brain_regions(df_pid['beryl_id'], channel_depths=df_pid['axial_um'].values,
                               brain_regions=regions, display=True, ax=axs[3], title='Aligned Beryl')
    plot_brain_regions(predictions_remap_beryl, channel_depths=df_pid['axial_um'].values,
                               brain_regions=regions, display=True, ax=axs[4], title='Classified')

    plot_probas(probas_beryl, legend=False, ax=axs[5])

    f.savefig(OUT_PATH.joinpath(f"{pid}.png"))

