from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ibllib.atlas import BrainRegions
import ephys_atlas.data

pd.set_option('use_inf_as_na', True)
x_list = []
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std',
          'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','psd_lfp','spike_count'] #1st this then swap to relative
x_list += ['peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val']
# x_list += ['atlas_id_planned']
# x_list += ['x_planned', 'y_planned', 'z_planned', 'peak_val']
regions = BrainRegions()

## data loading section
config = ephys_atlas.data.get_config()
# this path contains channels.pqt, clusters.pqt and raw_ephys_features.pqt
# path_features = "'/home/kcenia/Documents/atlas/2023_W14/"
path_features = "'/home/kcenia/Documents/atlas/2023_W34/"
# ephys_atlas.data.download_tables() can be used if features are not on disk
# df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_tables(path_features) #KB doesn't work for me
local_path = '/home/kcenia/Documents/atlas/2023_W34/' #KB added 19082023 
df_clusters = pd.read_parquet(local_path+'clusters.pqt') #KB added 19082023
df_channels = pd.read_parquet(local_path+'channels.pqt') #KB added 19082023
df_voltage = pd.read_parquet(local_path+'raw_ephys_features.pqt') #KB added 19082023
df_probes = pd.read_parquet(local_path+'probes.pqt') #KB added 19082023

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
df_voltage["cosmos_id"] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
df_voltage["beryl_id"] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')

""" to remove root and void """ 
# df_voltage = df_voltage.loc[~df_voltage['atlas_id'].isin(regions.acronym2id(['void', 'root']))]

#%% 
#plotting corr matrix
fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_voltage.corr(), annot=True, fmt='.2f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
# plt.savefig("/home/kcenia/Desktop/PEA/corr_matrix_psd_1.png", bbox_inches='tight', pad_inches=0.0) 

#adding the relative 
test = ['psd_delta', 'psd_theta',
       'psd_alpha', 'psd_beta', 'psd_gamma']
for column in test: 
    df_voltage[column+"_relative"] = df_voltage[column] - df_voltage["psd_lfp"] 
df_voltage_relative = df_voltage[['alpha_mean', 'alpha_std', 'spike_count', 'cloud_x_std', 'cloud_y_std',
       'cloud_z_std', 'peak_trace_idx', 'peak_time_idx', 'peak_val',
       'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val', 'rms_ap',
       'rms_lf', 'psd_delta_relative', 'psd_theta_relative', 'psd_alpha_relative',
       'psd_beta_relative', 'psd_gamma_relative', 'psd_lfp', 'x', 'y', 'z', 'acronym', 'atlas_id',
       'axial_um', 'lateral_um', 'histology', 'x_target', 'y_target',
       'z_target', 'atlas_id_target', 'cosmos_id', 'beryl_id']] 

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_voltage_relative.corr(), annot=True, fmt='.2f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
# plt.savefig("/home/kcenia/Desktop/PEA/corr_matrix_psd_1_relative.png", bbox_inches='tight', pad_inches=0.0) 


#%% 
#! ATTENTION 
#changing to the "relative" 
x_list = []
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std',
          'rms_lf', 'psd_delta_relative', 'psd_theta_relative', 'psd_alpha_relative', 'psd_beta_relative', 'psd_gamma_relative','psd_lfp','spike_count'] #1st this then swap to relative
x_list += ['peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val']
# df_voltage = df_voltage_relative 


#%%
"""
KB 19082023
based on: 
    paper-ephys-atlas/sources/examples
Selecting the data to train the model 
    > remove the 13 benchmark pids 
        > use df_voltage.loc["00a96dee-1e8b-44cc-9cc3-aca704d2b594"] 

""" 
# Training the model

n_trees = 30
max_depth = 25
max_leaf_nodes = 10000  # to be optimized! change the values until find when the model loses performance

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

idx_test = df_voltage.index.get_level_values(0).isin(benchmark_pids)
idx_train = ~idx_test



br = BrainRegions()

X_train = df_voltage.loc[idx_train, x_list].values 
y_train = df_voltage.loc[idx_train, "cosmos_id"].values  # we are training the model to already give the output as an acronym instead of an id

#%% 
""" added 31082023""" 
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X) 
X_train = scaler.transform(X_train)
X_test = df_voltage.loc[idx_test, x_list].values 
X_test = scaler.transform(X_test)


#%%
# Model

classifier=RandomForestClassifier(random_state=42, verbose=True, n_estimators=n_trees,
                                  max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

clf = classifier.fit(X_train, y_train) 

#%%

# Testing the model

X_test = df_voltage.loc[idx_test, x_list].values 
y_test = df_voltage.loc[idx_test, "cosmos_id"].values  # we are training the model to already give the output as an acronym instead of an id


predict_region = clf.predict(X_test)

score_model = clf.score(X_test, y_test)
print(score_model)
# 0.6484375 #21082023 cosmos with waveform features for 1 pid
# 0.5364583333333334
# 0.5669338677354709 same as above but 13pids 
# 0.5627254509018036 same as above (13 pids) but relative psds 
# 0.2565130260521042 13 pids beryl
# 0.251              13 pids beryl relative 
# 0.5050100200400801 13 pids cosmos no psds besides the psd_lfp 
# 0.5778095043850704 13 pids cosmos relative no root no void
# 0.26188048133795633 13 pids beryl relative no root no void 
# 0.20200400801603208 13 pids altas_id relative with root and void (6m25s) 





#%%
sns.set_theme(context='talk')
# forest_importances = pd.Series(cclas.feature_importances_, index=x_list)
# std = np.std([tree.feature_importances_ for tree in cclas.estimators_], axis=0)
forest_importances = pd.Series(clf.feature_importances_, index=x_list)
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()



#%%
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ephys_atlas.plots import plot_probas
from brainbox.ephys_plots import plot_brain_regions
from ibllib.atlas import BrainRegions

# Testing the model on
for pid in benchmark_pids:

    # pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
    df_pid = df_voltage.loc[pid].reset_index(drop=True)
    nc = df_pid.shape[0]
    X_pidtest = df_pid.loc[:, x_list].values
    score_pid = clf.score(X_pidtest, df_pid.loc[:, "cosmos_id"].values)  # todo put the target feature in parameter
    # we output both the most likely region and all of the probabilities for each region
    predict_region = clf.predict(X_pidtest)
    predict_regions_proba = pd.DataFrame(clf.predict_proba(X_pidtest).T)

    # this is the probability of the most likely region
    max_proba = np.max(predict_regions_proba.values, axis=0)

    # TODO aggregate per depth, not per channel !
    # we remap and aggregate probabilites
    predict_regions_proba['cosmos_aids'] = clf.classes_

    probas_cosmos = predict_regions_proba.groupby('cosmos_aids').sum().T


    cdepths = np.unique(df_channels['axial_um'])
    ccosmos = scipy.interpolate.interp1d(df_pid['axial_um'].values, predict_region, kind='nearest', fill_value="extrapolate")(cdepths)

    sns.set_theme()
    f, axs = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 8]})
    plot_brain_regions(df_pid['cosmos_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[0])
    plot_brain_regions(ccosmos, channel_depths=cdepths, brain_regions=regions, display=True, ax=axs[1], linewidth=0)

    plot_probas(probas_cosmos, legend=False, ax=axs[2])

    axs[1].set(yticks=[])
    axs[2].set(yticks=[],  xlabel='probability', title=f'Coarse parcellation, {score_pid}, {pid}')
    axs[2].grid(False)
    # f.savefig(f"/home/kcenia/Desktop/PEA/cumulativeproba_relative_{pid}.png")



#%% 
# Plotting Beryl 
for pid in benchmark_pids:

    # pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
    df_pid = df_voltage.loc[pid].reset_index(drop=True)
    nc = df_pid.shape[0]
    X_pidtest = df_pid.loc[:, x_list].values
    score_pid = clf.score(X_pidtest, df_pid.loc[:, "beryl_id"].values)  # todo put the target feature in parameter
    # we output both the most likely region and all of the probabilities for each region
    predict_region = clf.predict(X_pidtest)
    predict_regions_proba = pd.DataFrame(clf.predict_proba(X_pidtest).T)

    # this is the probability of the most likely region
    max_proba = np.max(predict_regions_proba.values, axis=0)

    # TODO aggregate per depth, not per channel !
    # we remap and aggregate probabilites
    predict_regions_proba['beryl_aids'] = clf.classes_

    probas_beryl = predict_regions_proba.groupby('beryl_aids').sum().T


    cdepths = np.unique(df_channels['axial_um'])
    cberyl = scipy.interpolate.interp1d(df_pid['axial_um'].values, predict_region, kind='nearest', fill_value="extrapolate")(cdepths)

    sns.set_theme()
    f, axs = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 8]})
    plot_brain_regions(df_pid['beryl_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[0])
    plot_brain_regions(cberyl, channel_depths=cdepths, brain_regions=regions, display=True, ax=axs[1], linewidth=0)

    plot_probas(probas_beryl, legend=False, ax=axs[2])

    axs[1].set(yticks=[])
    axs[2].set(yticks=[],  xlabel='probability', title=f'Beryl parcellation, {score_pid}, {pid}')
    axs[2].grid(False) 
    f.savefig(f"/home/kcenia/Desktop/PEA/cumulativeproba_Beryl_{pid}.png")
    # f.savefig(f"/home/kcenia/Desktop/PEA/cumulativeproba_relative_Beryl_{pid}.png") 

#%% 
#saved 
# Plotting Allen 
for pid in benchmark_pids:

    # pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
    df_pid = df_voltage.loc[pid].reset_index(drop=True)
    nc = df_pid.shape[0]
    X_pidtest = df_pid.loc[:, x_list].values
    score_pid = clf.score(X_pidtest, df_pid.loc[:, "atlas_id"].values)  # todo put the target feature in parameter
    # we output both the most likely region and all of the probabilities for each region
    predict_region = clf.predict(X_pidtest)
    predict_regions_proba = pd.DataFrame(clf.predict_proba(X_pidtest).T)

    # this is the probability of the most likely region
    max_proba = np.max(predict_regions_proba.values, axis=0)

    # TODO aggregate per depth, not per channel !
    # we remap and aggregate probabilites
    predict_regions_proba['atlas_aids'] = clf.classes_

    probas_cosmos = predict_regions_proba.groupby('atlas_aids').sum().T


    cdepths = np.unique(df_channels['axial_um'])
    ccosmos = scipy.interpolate.interp1d(df_pid['axial_um'].values, predict_region, kind='nearest', fill_value="extrapolate")(cdepths)

    sns.set_theme()
    f, axs = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 8]})
    plot_brain_regions(df_pid['atlas_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[0])
    plot_brain_regions(ccosmos, channel_depths=cdepths, brain_regions=regions, display=True, ax=axs[1], linewidth=0)

    plot_probas(probas_cosmos, legend=False, ax=axs[2])

    axs[1].set(yticks=[])
    axs[2].set(yticks=[],  xlabel='probability', title=f'Allen Atlas, {score_pid}, {pid}')
    axs[2].grid(False)
    f.savefig(f"/home/kcenia/Desktop/PEA/cumulativeproba_relative_Allen_{pid}.png") 

#%%
#%% 
#saved 
# Plotting Cosmos/Beryl from Allen training 
for pid in benchmark_pids:
    #pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
    df_pid = df_voltage.loc[pid].reset_index(drop=True)
    nc = df_pid.shape[0]
    X_pidtest = df_pid.loc[:, x_list].values
    score_pid = clf.score(X_pidtest, df_pid.loc[:, "atlas_id"].values)  # todo put the target feature in parameter
    # we output both the most likely region and all of the probabilities for each region
    predict_region = clf.predict(X_pidtest)
    predict_regions_proba = pd.DataFrame(clf.predict_proba(X_pidtest).T)

    # this is the probability of the most likely region
    max_proba = np.max(predict_regions_proba.values, axis=0)

    # TODO aggregate per depth, not per channel !
    # we remap and aggregate probabilites 
    predict_region = regions.remap(predict_region, source_map='Allen', target_map='Cosmos') 
    #predict_region = regions.remap(predict_region, source_map='Allen', target_map='Beryl') 

    predict_regions_proba['atlas_aids'] = clf.classes_ 

    predict_regions_proba['cosmos_aids'] = regions.remap(predict_regions_proba["atlas_aids"], source_map='Allen', target_map='Cosmos')
    probas_cosmos = predict_regions_proba.groupby('cosmos_aids').sum().T 
    probas_cosmos = probas_cosmos[:-1]
    # predict_regions_proba['beryl_aids'] = regions.remap(predict_regions_proba["atlas_aids"], source_map='Allen', target_map='Beryl')
    # probas_cosmos = predict_regions_proba.groupby('beryl_aids').sum().T 
    # probas_cosmos = probas_cosmos[:-1]


    cdepths = np.unique(df_channels['axial_um'])
    ccosmos = scipy.interpolate.interp1d(df_pid['axial_um'].values, predict_region, kind='nearest', fill_value="extrapolate")(cdepths)

    sns.set_theme()
    f, axs = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 8]})
    plot_brain_regions(df_pid['cosmos_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[0]) 
    # plot_brain_regions(df_pid['beryl_id'], channel_depths=df_pid['axial_um'].values, brain_regions=regions, display=True, ax=axs[0])
    plot_brain_regions(ccosmos, channel_depths=cdepths, brain_regions=regions, display=True, ax=axs[1], linewidth=0)

    plot_probas(probas_cosmos, legend=False, ax=axs[2])

    axs[1].set(yticks=[])
    axs[2].set(yticks=[],  xlabel='probability', title=f'Allen Atlas, {score_pid}, {pid}')
    axs[2].grid(False)
    # f.savefig(f"/home/kcenia/Desktop/PEA/cumulativeproba_relative_Cosmos_from_Allen_{pid}.png") 
    # f.savefig(f"/home/kcenia/Desktop/PEA/cumulativeproba_relative_Beryl_from_Allen_{pid}.png") 


#%%




























#%% NOT USED FOR NOW 21082023
#
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)
stratify = df_voltage.index.get_level_values('pid')
kwargs = {'n_estimators': 30, 'max_depth': 25, 'max_leaf_nodes': 10000, 'random_state': 420}


# It would also be informative to check that the target location error increases as a function of depth.  Can just scatterplot error vs depth (instead of histogram()
# June 6th practice / June 15th U19 talk


def decode(X, scaler, aids, classifier=None, save_path=None, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(X, aids, stratify=stratify, random_state=875)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = classifier.fit(X_train, y_train)
    score_model = clf.score(X_test, y_test)
    y_null = aids[np.random.randint(0, aids.size - 1, y_test.size)]
    score_null = clf.score(X_test, y_null)
    return clf, score_model, score_null




cclas, cs, csn = decode(X, scaler, aids_cosmos, classifier=RandomForestClassifier(verbose=True, **kwargs), stratify=stratify)
bclas, bs, bsn = decode(X, scaler, aids_beryl, classifier=RandomForestClassifier(verbose=True, **kwargs), stratify=stratify)
print(bs, bsn, cs, csn)
#
sns.set_theme(context='talk')
# forest_importances = pd.Series(cclas.feature_importances_, index=x_list)
# std = np.std([tree.feature_importances_ for tree in cclas.estimators_], axis=0)
forest_importances = pd.Series(bclas.feature_importances_, index=x_list)
std = np.std([tree.feature_importances_ for tree in bclas.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# 0.40807415084013554 0.10099092909791683 0.5734580048673134 0.12062320241269693
# 0.44346114882567333 0.09820794373478965 0.6149000337684416 0.12400004657716089
# 0.5216991348292365 0.09059257792941232 0.659602463931811 0.12369729503138137
# 0.4063157160948519 0.09833741910748103 0.5697970280349943 0.12273588084256021 #KB 19082023

##
# auto-encoder or contrastive learning
# take your learned features and model spikes
# phase / frequency / amplitude
accuracy_score(df_voltage['atlas_id'], df_voltage['atlas_id_planned'])  # 0.12
accuracy_score(df_voltage['beryl_id'], regions.remap(df_voltage['atlas_id_planned'], source_map='Allen', target_map='Beryl'))  # 0.19 - 0.91
accuracy_score(df_voltage['cosmos_id'], regions.remap(df_voltage['atlas_id_planned'], source_map='Allen', target_map='Cosmos'))  # 0.44 - 0.95






























































































































# %% 
#--------------------------------------------------------------------------------------------------------------------------
#                                                     WITH NEURAL NETS 
#--------------------------------------------------------------------------------------------------------------------------
""" WITH NEURAL NETS """ 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ibllib.atlas import BrainRegions
import ephys_atlas.data
pd.set_option('use_inf_as_na', True)
x_list = []
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std',
          'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','psd_lfp','spike_count'] #1st this then swap to relative

x_list += ['peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val']
# x_list += ['atlas_id_planned']
# x_list += ['x_planned', 'y_planned', 'z_planned', 'peak_val']

regions = BrainRegions()
## data loading section

config = ephys_atlas.data.get_config()
# this path contains channels.pqt, clusters.pqt and raw_ephys_features.pqt
# path_features = "'/home/kcenia/Documents/atlas/2023_W14/"

path_features = "'/home/kcenia/Documents/atlas/2023_W34/"
# ephys_atlas.data.download_tables() can be used if features are not on disk
# df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_tables(path_features) #KB doesn't work for me

local_path = '/home/kcenia/Documents/atlas/2023_W34/' #KB added 19082023 
df_clusters = pd.read_parquet(local_path+'clusters.pqt') #KB added 19082023
df_channels = pd.read_parquet(local_path+'channels.pqt') #KB added 19082023
df_voltage = pd.read_parquet(local_path+'raw_ephys_features.pqt') #KB added 19082023
df_probes = pd.read_parquet(local_path+'probes.pqt') #KB added 19082023
df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
df_voltage["cosmos_id"] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
df_voltage["beryl_id"] = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')
""" to remove root and void """ 
# df_voltage = df_voltage.loc[~df_voltage['atlas_id'].isin(regions.acronym2id(['void', 'root']))]
#%% 
#plotting corr matrix

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_voltage.corr(), annot=True, fmt='.2f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)

ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
# plt.savefig("/home/kcenia/Desktop/PEA/corr_matrix_psd_1.png", bbox_inches='tight', pad_inches=0.0) 
#adding the relative 

test = ['psd_delta', 'psd_theta',
       'psd_alpha', 'psd_beta', 'psd_gamma']

for column in test: 
    df_voltage[column+"_relative"] = df_voltage[column] - df_voltage["psd_lfp"] 

df_voltage_relative = df_voltage[['alpha_mean', 'alpha_std', 'spike_count', 'cloud_x_std', 'cloud_y_std',
       'cloud_z_std', 'peak_trace_idx', 'peak_time_idx', 'peak_val',
       'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val', 'rms_ap',
       'rms_lf', 'psd_delta_relative', 'psd_theta_relative', 'psd_alpha_relative',
       'psd_beta_relative', 'psd_gamma_relative', 'psd_lfp', 'x', 'y', 'z', 'acronym', 'atlas_id',
       'axial_um', 'lateral_um', 'histology', 'x_target', 'y_target',
       'z_target', 'atlas_id_target', 'cosmos_id', 'beryl_id']] 

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_voltage_relative.corr(), annot=True, fmt='.2f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)

ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
# plt.savefig("/home/kcenia/Desktop/PEA/corr_matrix_psd_1_relative.png", bbox_inches='tight', pad_inches=0.0) 
#changing to the "relative" 

x_list = []
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std',
          'rms_lf', 'psd_delta_relative', 'psd_theta_relative', 'psd_alpha_relative', 'psd_beta_relative', 'psd_gamma_relative','psd_lfp','spike_count'] #1st this then swap to relative

x_list += ['peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val'] 
df_voltage = df_voltage_relative 





#%%
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

idx_test = df_voltage.index.get_level_values(0).isin(benchmark_pids)
idx_train = ~idx_test
br = BrainRegions()
X_train = df_voltage.loc[idx_train, x_list].values 
y_train = df_voltage.loc[idx_train, "cosmos_id"].values  # we are training the model to already give the output as an acronym instead of an id
#%% 

""" added 31082023""" 
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X) 
X_train = scaler.transform(X_train)
X_test = df_voltage.loc[idx_test, x_list].values 
X_test = scaler.transform(X_test)
y_test = df_voltage.loc[idx_test, "cosmos_id"].values  # we are training the model to already give the output as an acronym instead of an id




#%%
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(250, 125), random_state=1)
clf.fit(X_train, y_train) 

predict_region = clf.predict(X_test)
score_model = clf.score(X_test, y_test)
print(score_model) 
