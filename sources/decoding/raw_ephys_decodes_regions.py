from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ibllib.atlas import BrainRegions

regions = BrainRegions()
LOCAL_DATA_PATH = Path("/datadisk/Data/aggregates/bwm")
OUT_PATH = Path("/datadisk/gdrive/2022/08_ephys_atlas_features")

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


def decode(X, scaler, aids, classifier=None, save_path=None):
    if classifier is None:
        classifier = MLPClassifier(random_state=1, max_iter=300, verbose=True)
    X_train, X_test, y_train, y_test = train_test_split(X, aids, stratify=aids)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = classifier.fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    score_model = clf.score(X_test, y_test)
    y_null = aids_cosmos[np.random.randint(0, aids.size - 1, y_test.size)]
    score_null = clf.score(X_test, y_null)
    return clf, score_model, score_null


 ## %%
from sklearn.ensemble import RandomForestClassifier
cclas, cs, csn = decode(X, scaler, aids_cosmos, classifier=RandomForestClassifier(verbose=True))
bclas, bs, bsn = decode(X, scaler, aids_beryl, classifier=RandomForestClassifier(verbose=True))
print(bs, bsn, cs, csn)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme('paper')
# forest_importances = pd.Series(cclas.feature_importances_, index=x_list)
# std = np.std([tree.feature_importances_ for tree in cclas.estimators_], axis=0)
forest_importances = pd.Series(bclas.feature_importances_, index=x_list)
std = np.std([tree.feature_importances_ for tree in bclas.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# 0.5168615851641402 0.03773181030959933 0.6357002471346008 0.11985055167448287

## %% Decode Beryl and Cosmos using Neural nets
_, bs, bsn = decode(X, scaler, aids_beryl)
_, cs, csn = decode(X, scaler, aids_cosmos)
print(bs, bsn, cs, csn)
# 0.3504251882698632, 0.03403452100644107, 0.542723150868863, 0.1213878456479013

## %
from sklearn.naive_bayes import GaussianNB
_, cs, csn = decode(X, scaler, aids_cosmos, classifier=GaussianNB())
_, bs, bsn = decode(X, scaler, aids_beryl, classifier=GaussianNB())
print(bs, bsn, cs, csn)
# 0.08093171690439588 0.0002140535912354784 0.31541769639416994 0.1029013991321099
