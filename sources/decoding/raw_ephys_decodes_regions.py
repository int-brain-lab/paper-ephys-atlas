from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from iblatlas.atlas import BrainRegions
import ephys_atlas.data

pd.set_option("use_inf_as_na", True)
x_list = []
# x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std',
#           'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma','spike_count']
# x_list += ['peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val']
# x_list += ['atlas_id_planned']
x_list += ["x_planned", "y_planned", "z_planned", "peak_val"]
regions = BrainRegions()

## data loading section
config = ephys_atlas.data.get_config()
# this path contains channels.pqt, clusters.pqt and raw_ephys_features.pqt
path_features = Path(config["paths"]["features"]).joinpath(config["decoding"]["tag"])

df_voltage, df_clusters, df_channels = ephys_atlas.data.load_tables(path_features)
df_voltage = pd.merge(
    df_voltage, df_channels, left_index=True, right_index=True
).dropna()

aids_cosmos = regions.remap(
    df_voltage["atlas_id"], source_map="Allen", target_map="Cosmos"
)
aids_beryl = regions.remap(
    df_voltage["atlas_id"], source_map="Allen", target_map="Beryl"
)
df_voltage["beryl_id"] = aids_beryl
df_voltage["cosmos_id"] = aids_cosmos

#


X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)
stratify = df_voltage.index.get_level_values("pid")
kwargs = {
    "n_estimators": 30,
    "max_depth": 25,
    "max_leaf_nodes": 10000,
    "random_state": 420,
}

# auto-encoder or contrastive learning
# take your learned features and model spikes
# phase / frequency / amplitude
accuracy_score(df_voltage["atlas_id"], df_voltage["atlas_id_planned"])  # 0.12
accuracy_score(
    df_voltage["beryl_id"],
    regions.remap(
        df_voltage["atlas_id_planned"], source_map="Allen", target_map="Beryl"
    ),
)  # 0.19 - 0.91
accuracy_score(
    df_voltage["cosmos_id"],
    regions.remap(
        df_voltage["atlas_id_planned"], source_map="Allen", target_map="Cosmos"
    ),
)  # 0.44 - 0.95


# It would also be informative to check that the target location error increases as a function of depth.  Can just scatterplot error vs depth (instead of histogram()
# June 6th practice / June 15th U19 talk


def decode(X, scaler, aids, classifier=None, save_path=None, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, aids, stratify=stratify, random_state=875
    )
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = classifier.fit(X_train, y_train)
    score_model = clf.score(X_test, y_test)
    y_null = aids[np.random.randint(0, aids.size - 1, y_test.size)]
    score_null = clf.score(X_test, y_null)
    return clf, score_model, score_null


cclas, cs, csn = decode(
    X,
    scaler,
    aids_cosmos,
    classifier=RandomForestClassifier(verbose=True, **kwargs),
    stratify=stratify,
)
bclas, bs, bsn = decode(
    X,
    scaler,
    aids_beryl,
    classifier=RandomForestClassifier(verbose=True, **kwargs),
    stratify=stratify,
)
print(bs, bsn, cs, csn)
#
sns.set_theme(context="talk")
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


##
