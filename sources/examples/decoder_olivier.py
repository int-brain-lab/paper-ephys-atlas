#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:46:19 2023
@author: Joana Catarino, Kcénia Bougrova, Olivier Winter
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
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.interpolate
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier

from ephys_atlas.plots import plot_probas_df
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import BrainRegions
import ephys_atlas.data
regions = BrainRegions()
BENCHMARK = True
SCALE = False

label = '2023_W34'

LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
# FOLDER_GDRIVE = Path("/mnt/s1/ephys-atlas-decoding")


benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724',
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

df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_tables(LOCAL_DATA_PATH.joinpath(label), verify=True)
df_voltage.replace([np.inf, -np.inf], np.nan, inplace=True)


df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
# df_voltage = df_voltage.loc[df_voltage['histology'].isin(['alf', 'resolved']), :]
df_voltage['cosmos_id'] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
df_voltage['beryl_id'] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')

df_voltage = df_voltage.loc[~df_voltage['atlas_id'].isin(regions.acronym2id(['void']))]
# for feat in ['rms_ap', 'rms_lf']:
#     df_voltage[feat] = 20 * np.log10(df_voltage[feat])
print(f"{df_voltage.shape[0]} channels")
# selection and scaling of features
# old versions
# x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'spike_count', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
# x_list += ['peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val']

x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'spike_count', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
x_list += ['peak_time_secs', 'peak_val', 'trough_time_secs', 'trough_val', 'tip_time_secs', 'tip_val']
x_list += ['polarity', 'depolarisation_slope', 'repolarisation_slope', 'recovery_time_secs', 'recovery_slope']
x_list += ['psd_lfp_csd']
# Training the model
# kwargs = {'n_estimators': 30, 'max_depth': 25, 'max_leaf_nodes': 10000, 'random_state': 420}
kwargs = {'n_estimators': 30, 'max_depth': 25, 'max_leaf_nodes': 10000, 'random_state': 420, 'n_jobs': -1, 'criterion': 'entropy'}

if BENCHMARK:
    test_idx = df_voltage.index.get_level_values(0).isin(benchmark_pids)
    train_idx = ~test_idx

x_train = df_voltage.loc[train_idx, x_list].values
x_test = df_voltage.loc[test_idx, x_list].values

if SCALE:
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_test = scaler.transform(x_test)
    x_train = scaler.transform(x_train)


for mapping in ['cosmos_id', 'beryl_id', 'atlas_id']:
    y_test = df_voltage.loc[test_idx, mapping]
    y_train = df_voltage.loc[train_idx, mapping]
    classifier = RandomForestClassifier(verbose=True, **kwargs)
    clf = classifier.fit(x_train, y_train)
    y_null = np.random.choice(df_voltage.loc[train_idx, mapping], y_test.size)
    print(mapping, clf.score(x_test, y_test), clf.score(x_test, y_null))

    # feature_imp[i, :] = clf.feature_importances_
    break
    if mapping == 'atlas_id':
        predict_regions_proba = pd.DataFrame(clf.predict_proba(x_test).T)
        predict_regions_proba['beryl_aids'] = regions.remap(clf.classes_, source_map='Allen', target_map='Beryl')
        predict_regions_proba['cosmos_aids'] = regions.remap(clf.classes_, source_map='Allen', target_map='Cosmos')
        probas_beryl = predict_regions_proba.groupby('beryl_aids').sum().drop(columns='cosmos_aids').T
        probas_cosmos = predict_regions_proba.groupby('cosmos_aids').sum().drop(columns='beryl_aids').T
        sb = sklearn.metrics.accuracy_score(regions.remap(y_test, source_map='Allen', target_map='Beryl'), probas_beryl.columns[np.argmax(probas_beryl.values, axis=1)].values)
        sc = sklearn.metrics.accuracy_score(regions.remap(y_test, source_map='Allen', target_map='Cosmos'), probas_cosmos.columns[np.argmax(probas_cosmos.values, axis=1)].values)
        print(f"remapped scores: beryl {sb}, cosmos {sc}")


import seaborn as sns
sns.set_context('paper')
f, ax = plt.subplots(1, 1)
ax.bar(x_list, clf.feature_importances_)
ax.set(ylabel='Feature importance')
ax.set_xticklabels(x_list, rotation=90)
# f.tight_layout()

# removed both void and root 2023_W33
# beryl_id 0.25095227425498545 0.03136903428187318
# cosmos_id 0.6287250728209725 0.1382478153708268

# removed only void 2023_W33
# beryl_id 0.25829432118868306 0.09200081416649705
# cosmos_id 0.5583146753511093 0.11764705882352941

# removed only void 2023_W34 same features
# beryl_id 0.2589049460614696 0.0789741502137187
# cosmos_id 0.5713413393038876 0.1245674740484429

# removed only void 2023_W34 add new spike features
# beryl_id 0.2597191125585182 0.0964787299002646

# cosmos_id 0.5570934256055363 0.11683289232648077
# atlas_id 0.18318746183594545 0.02177895379605129
# remapped scores: beryl 0.2475066151027885, cosmos 0.5308365560757174



## %%


## %%
# Testing the model on
for pid in Benchmark_pids:
    pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
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


# .//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#     cdepths = np.unique(df_channels['axial_um'])
#     ccosmos = scipy.interpolate.interp1d(df_pid['axial_um'].values, predictions_remap_cosmos, kind='nearest', fill_value="extrapolate")(cdepths)
#     cberyl = scipy.interpolate.interp1d(df_pid['axial_um'].values, predictions_remap_beryl, kind='nearest', fill_value="extrapolate")(cdepths)
#
#     sns.set_theme()
#     f, axs = plt.subplots(1, 7, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 8, 2, 1, 1, 8]})
#     plot_brain_regions(df_pid['cosmos_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[0])
#     plot_brain_regions(ccosmos, channel_depths=cdepths, brain_regions=regions, display=True, ax=axs[1], linewidth=0)
    plot_probas_df(probas_cosmos, legend=False, ax=axs[2])
    plot_brain_regions(df_pid['beryl_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[4])
    plot_brain_regions(cberyl, channel_depths=cdepths, brain_regions=regions, display=True, ax=axs[5], linewidth=0)
    plot_probas_df(probas_beryl, legend=False, ax=axs[6])
    axs[3].set(visible=False)
    axs[1].set(yticks=[])
    axs[2].set(yticks=[],  xlabel='probability', title='Coarse parcellation')
    axs[2].grid(False)
    axs[5].set(yticks=[])
    axs[6].set(xlabel='probability', title='Fine parcellation', ylabel='electrode distance from tip (mm)')
    axs[6].grid(False)
    axs[6].yaxis.set_ticks_position('right')
    axs[6].yaxis.set_label_position('right')
    axs[6].yaxis.set_major_formatter(lambda x, pos: f"{x / 100 :1.1f}")
    f.savefig(OUT_PATH.joinpath(f"{pid}.png"))

    # plt.close(f)
    break

## %%
    # from matplotlib.container import BarContainer
    # [c.get_children()[0].set_linewidth(0) for c in axs[1].containers if isinstance(c, BarContainer)]
    # [c.get_children()[0].set_linewidth(0) for c in axs[4].containers if isinstance(c, BarContainer)]