# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:05:06 2023

@author: Joana
"""

# %% Benchmark pids
"""
1) KS046          2020.12.03   P00    1a276285-8b0e-4cc9-9f0a-a3a002978724
2) SWC_043        2020.09.20   P00    1e104bf4-7a24-4624-a5b2-c2c8289c0de7
3) CSH_ZAD_029    2020.09.17   PO1    5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e
4) NYU-40         2021.04.15   P01    f7766ce-8e2e-410c-9195-6bf089fea4fd
5) SWC_043        2020.09.21   P01    6638cfb3-3831-4fc2-9327-194b76cf22e1
6) SWC_038        2020.07.30   P01    749cb2b7-e57e-4453-a794-f6230e4d0226
7) KS096          2022.06.17   P00    d7ec0892-0a6c-4f4f-9d8f-72083692af5c
8) SWC_043        2020.09.21   P00    da8dfec1-d265-44e8-84ce-6ae9c109b8bd
9) DY_016         2020.09.12   P00    dab512bd-a02d-4c1f-8dbc-9155a163efc0
10) ZM_2241       2020.01.27   P00    dc7e9403-19f7-409f-9240-05ee57cb7aea
11) ZM_1898       2019.12.10   P00    e8f9fba4-d151-4b00-bee7-447f0f3e752c
12) UCLA033       022.02.15    P01    eebcaf65-7fa4-4118-869d-a084e84530e2
13) CSH_ZAD_025   2020.08.03   P01    fe380793-8035-414e-b000-09bfe5ece92a

"""
# %%

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import scipy.interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from ephys_atlas.plots import plot_probas_df
from brainbox.ephys_plots import plot_brain_regions
from ephys_atlas.data import download_tables, compute_depth_dataframe, load_tables
from iblatlas.atlas import BrainRegions
from iblutil.numerical import ismember
from joblib import dump

from one.api import ONE
from pathlib import Path
from one.remote import aws

from brainbox.ephys_plots import plot_brain_regions
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.container import BarContainer

regions = BrainRegions()
BENCHMARK = True
SCALE = True

# %% Download tables - Only needs to be done once
"""
LABEL = '2022_W34'
LABEL = '2023_W14'
FOLDER_GDRIVE = Path("C:/Users/Asus/int-brain-lab/ephys_atlas/tables")
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_raw_features, df_clusters, df_channels = download_tables(label=LABEL, local_path=FOLDER_GDRIVE, one=one)
df_depths = compute_depth_dataframe(df_raw_features, df_clusters, df_channels)
df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)

## %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')
"""
# %%

LOCAL_DATA_PATH = Path("C:/Users/Asus/int-brain-lab/ephys_atlas/tables")
OUT_PATH = Path("C:/Users/Asus/int-brain-lab/ephys_atlas/decoding/plots")
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="local")

df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath("channels.pqt"))
df_channels.index.rename("channel", level=1, inplace=True)
df_raw_features1 = pd.read_parquet(LOCAL_DATA_PATH.joinpath("raw_ephys_features.pqt"))

pd.set_option("use_inf_as_na", True)
df_raw_features1 = pd.merge(
    df_raw_features1, df_channels, left_index=True, right_index=True
).dropna()

# remapping the lowest level of channel location to higher levels mappings
df_raw_features1["cosmos_id"] = regions.remap(
    df_raw_features1["atlas_id"], source_map="Allen", target_map="Cosmos"
)
df_raw_features1["beryl_id"] = regions.remap(
    df_raw_features1["atlas_id"], source_map="Allen", target_map="Beryl"
)


# use lines 96 to 103 to sort insertions to include in the training dataset based on histology status
# To include only resolved alignement in the training dataset
df_raw_features = df_raw_features1.loc[
    (df_raw_features1.histology == "resolved") | (df_raw_features1.histology == "alf")
]

# To include resolved and aligned probes in the training dataset
# df_raw_features = df_raw_features1.loc[(df_raw_features1.histology == 'resolved') | (df_raw_features1.histology == 'alf') | (df_raw_features1.histology == 'aligned')]

# To include only the traced probes in the training dataset
# df_raw_features = df_raw_features1.loc[(df_raw_features1.histology == 'traced')]

# to elimante void and root from the dataset
a, _ = ismember(
    df_raw_features["cosmos_id"], regions.acronym2id(["void", "root"])
)  # To exclude void and root when training the model
df_raw_features = df_raw_features.loc[~a]
# df_raw_features = df_raw_features.reset_index(drop=True)

# selection and scaling of features
x_list = [
    "rms_ap",
    "alpha_mean",
    "alpha_std",
    "spike_count",
    "cloud_x_std",
    "cloud_y_std",
    "cloud_z_std",
    "rms_lf",
    "psd_delta",
    "psd_theta",
    "psd_alpha",
    "psd_beta",
    "psd_gamma",
]  # Use this only to give 13 features as input to the model
x_list += [
    "peak_time_idx",
    "peak_val",
    "trough_time_idx",
    "trough_val",
    "tip_time_idx",
    "tip_val",
]  # Add this to give a total of 22 features as an input to the model

# Training the model
kwargs = {
    "n_estimators": 30,
    "max_depth": 25,
    "max_leaf_nodes": 10000,
    "random_state": 420,
}


# %%
benchmark_pids = [
    "1a276285-8b0e-4cc9-9f0a-a3a002978724",
    "1e104bf4-7a24-4624-a5b2-c2c8289c0de7",
    "5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e",
    "5f7766ce-8e2e-410c-9195-6bf089fea4fd",
    "6638cfb3-3831-4fc2-9327-194b76cf22e1",
    "749cb2b7-e57e-4453-a794-f6230e4d0226",
    "d7ec0892-0a6c-4f4f-9d8f-72083692af5c",
    "da8dfec1-d265-44e8-84ce-6ae9c109b8bd",
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0",
    "dc7e9403-19f7-409f-9240-05ee57cb7aea",
    "e8f9fba4-d151-4b00-bee7-447f0f3e752c",
    "eebcaf65-7fa4-4118-869d-a084e84530e2",
    "fe380793-8035-414e-b000-09bfe5ece92a",
]

# %% Training the model

if BENCHMARK:
    test_idx = df_raw_features.index.get_level_values(0).isin(benchmark_pids)
    train_idx = ~test_idx
else:
    pass
    # shuffled_pids = np.random.permutation(list(df_raw_features.index.get_level_values(0).unique()))
    # tlast = int(np.round(shuffled_pids.size / 4))
    # test_pids = shuffled_pids[:tlast]
    # train_pids = shuffled_pids[tlast:]

x_train = df_raw_features.loc[train_idx, x_list].values
x_test = df_raw_features.loc[test_idx, x_list].values

if SCALE:
    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)
    x_train = scaler.transform(x_train)

clfs = []

for mapping in ["beryl_id", "cosmos_id", "atlas_id"]:
    y_test = df_raw_features.loc[test_idx, mapping]
    y_train = df_raw_features.loc[train_idx, mapping]
    classifier = RandomForestClassifier(verbose=True, **kwargs)
    clf = classifier.fit(x_train, y_train)
    clfs.append(clf)
    y_null = np.random.choice(df_raw_features.loc[train_idx, mapping], y_test.size)
    print(mapping, clf.score(x_test, y_test), clf.score(x_test, y_null))

# %% Testing the model

df_voltage = df_raw_features.reset_index(["pid", "channel"])

clf = clfs[0]
predict_region_pids = []


for pid in benchmark_pids:
    df_pid = df_voltage[df_voltage.pid == pid]
    df_pid = df_pid.reset_index(drop=True)
    nc = df_pid.shape[0]
    X_test = df_pid.loc[:, x_list].values

    # we output both the most likely region and all of the probabilities for each region
    if SCALE:
        X_test = scaler.transform(X_test)

    predict_region = clf.predict(X_test)
    predict_region_pids.append(predict_region)
    predict_regions_proba = pd.DataFrame(clf.predict_proba(X_test).T)

    # this is the probability of the most likely region
    max_proba = np.max(predict_regions_proba.values, axis=0)

    # TODO aggregate per depth, not per channel !

    # we remap and aggregate probabilites
    predict_regions_proba["beryl_aids"] = regions.remap(
        clf.classes_, source_map="Beryl", target_map="Beryl"
    )
    predict_regions_proba["cosmos_aids"] = regions.remap(
        clf.classes_, source_map="Beryl", target_map="Cosmos"
    )
    probas_beryl = (
        predict_regions_proba.groupby("beryl_aids").sum().drop(columns="cosmos_aids").T
    )
    probas_cosmos = (
        predict_regions_proba.groupby("cosmos_aids").sum().drop(columns="beryl_aids").T
    )

    predictions_remap_cosmos = regions.remap(
        predict_region, source_map="Beryl", target_map="Cosmos"
    )
    predictions_remap_beryl = regions.remap(
        predict_region, source_map="Beryl", target_map="Beryl"
    )

    cdepths = np.unique(df_channels["axial_um"])
    ccosmos = scipy.interpolate.interp1d(
        df_pid["axial_um"].values,
        predictions_remap_cosmos,
        kind="nearest",
        fill_value="extrapolate",
    )(cdepths)
    cberyl = scipy.interpolate.interp1d(
        df_pid["axial_um"].values,
        predictions_remap_beryl,
        kind="nearest",
        fill_value="extrapolate",
    )(cdepths)

    # Plot real/predicted probes + cumulative probability
    sns.set_theme()
    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
        1,
        7,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1.2, 1.2, 4.8, 1.5, 1.2, 1.2, 4.8]},
        dpi=500,
    )

    plot_brain_regions(
        df_pid["cosmos_id"],
        channel_depths=df_pid["axial_um"].values,
        brain_regions=regions,
        display=True,
        ax=ax1,
        title="Aligned Cosmos",
    )
    ax1.set_title("Aligned Cosmos", fontsize=10, pad=11)

    plot_brain_regions(
        predictions_remap_cosmos,
        channel_depths=df_pid["axial_um"].values,
        brain_regions=regions,
        display=True,
        ax=ax2,
        title="Predicted",
    )
    ax2.set_title("Predicted", fontsize=10, pad=11)
    ax2.set(yticks=[])

    plot_probas_df(probas_cosmos, legend=False, ax=ax3)
    ax3.tick_params(labelsize=9)
    ax3.set_title("Cumulative Probability", fontsize=10, pad=11)
    ax3.yaxis.set_ticks_position("right")
    ax3.grid(False)

    ax4.set(visible=False)

    plot_brain_regions(
        df_pid["beryl_id"],
        channel_depths=df_pid["axial_um"].values,
        brain_regions=regions,
        display=True,
        ax=ax5,
        title="Aligned Beryl",
    )
    ax5.set_title("Aligned Beryl", fontsize=10, pad=11)

    plot_brain_regions(
        predictions_remap_beryl,
        channel_depths=df_pid["axial_um"].values,
        brain_regions=regions,
        display=True,
        ax=ax6,
        title="Predicted",
    )
    ax6.set_title("Predicted", fontsize=10, pad=11)
    ax6.set(yticks=[])

    plot_probas_df(probas_beryl, legend=False, ax=ax7)
    ax7.tick_params(labelsize=9)
    ax7.set_title("Cumulative Probability", fontsize=10, pad=11)
    ax7.set_ylabel("Channel number", fontsize=10, labelpad=12)
    ax7.yaxis.set_ticks_position("right")
    ax7.yaxis.set_label_position("right")
    ax7.grid(False)

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.30, hspace=0.8
    )

    # Remove white stripes on the probe plots
    [
        c.get_children()[0].set_linewidth(0)
        for c in ax2.containers
        if isinstance(c, BarContainer)
    ]
    [
        c.get_children()[0].set_linewidth(0)
        for c in ax6.containers
        if isinstance(c, BarContainer)
    ]

    f.savefig(OUT_PATH.joinpath(f"{pid}.png"))
    f.savefig(OUT_PATH.joinpath(f"{pid}.pdf"))

    print("Done for " f"{pid}")
