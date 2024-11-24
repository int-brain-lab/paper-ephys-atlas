import pandas as pd
import numpy as np

from pathlib import Path
from one.remote import aws
from one.api import ONE

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from random import shuffle

import matplotlib.pyplot as plt


def get_data():
    LOCAL_DATA_PATH = Path("/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/atlas_data")

    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="online")
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_folder(
        "aggregates/bwm", LOCAL_DATA_PATH, s3=s3, bucket_name=bucket_name
    )

    df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath("clusters.pqt"))
    df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath("probes.pqt"))
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath("channels.pqt"))
    df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath("depths.pqt"))
    df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath("raw_ephys_features.pqt"))

    df_voltage = pd.merge(
        df_voltage, df_channels, left_index=True, right_index=True
    ).dropna()

    return df_voltage


def regress(scaling=True, shuf=False):
    df_voltage = get_data()

    x_list = [
        "rms_ap",
        "alpha_mean",
        "alpha_std",
        "spike_rate",
        "cloud_x_std",
        "cloud_y_std",
        "cloud_z_std",
        "rms_lf",
        "psd_delta",
        "psd_theta",
        "psd_alpha",
        "psd_beta",
        "psd_gamma",
    ]

    x_list = np.sort(x_list)
    X = df_voltage.loc[:, x_list].values
    y = df_voltage.loc[:, ["x", "y", "z"]].values

    print(X.shape, "samples x input_features")
    print(y.shape, "samples x target_features")

    if shuf:
        shuffle(y)

    # cross validation
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True)
    fold = 0

    coefs = []
    scores = dict(zip(["x", "y", "z"], [[], [], []]))
    for tra, tes in kf.split(X):
        X_tra = X[tra]
        X_tes = X[tes]
        y_tra = y[tra]
        y_tes = y[tes]

        if scaling:
            scaler = StandardScaler()
            scaler.fit(X_tra)
            X_tra = scaler.transform(X_tra)
            X_tes = scaler.transform(X_tes)

        ci = []
        for i in range(y.shape[-1]):
            reg = SGDRegressor(loss="huber", early_stopping=True).fit(
                X_tra, y_tra[:, i]
            )

            ci.append(reg.coef_)

            sco = np.round(reg.score(X_tes, y_tes[:, i]), 2)
            scores[["x", "y", "z"][i]].append(sco)
            print(["x", "y", "z"][i], ", fold: ", fold, ", score: ", sco)

        coefs.append(ci)
        fold += 1

    return scores, np.array(coefs)


def plot_scores(scores):
    fig, ax = plt.subplots()
    ax.bar(
        range(len(scores)),
        [np.mean(scores[x]) for x in scores],
        yerr=[np.std(scores[x]) for x in scores],
    )
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([x for x in scores])
    ax.set_xlabel("channel position")
    ax.set_ylabel("accuracy, r**2")


def plot_coefs(coefs):
    # coefs = regress()

    coefs = np.abs(coefs)

    x_list = [
        "rms_ap",
        "alpha_mean",
        "alpha_std",
        "spike_rate",
        "cloud_x_std",
        "cloud_y_std",
        "cloud_z_std",
        "rms_lf",
        "psd_delta",
        "psd_theta",
        "psd_alpha",
        "psd_beta",
        "psd_gamma",
    ]
    x_list = np.sort(x_list)

    ys = ["x", "y", "z"]

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)

    for i in range(len(ys)):
        # mean and std across folds
        y = coefs.mean(axis=0)[i]
        yerr = coefs.std(axis=0)[i]
        axs[i].bar(range(len(x_list)), y, yerr=yerr)
        axs[i].set_ylabel(f"weights to {ys[i]}")
        axs[i].set_xlabel("features")
        axs[i].set_xticks(range(len(x_list)))
        axs[i].set_xticklabels(x_list, rotation=90)

    fig.tight_layout()
