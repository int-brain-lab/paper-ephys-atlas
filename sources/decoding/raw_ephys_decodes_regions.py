from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ibllib.atlas import BrainRegions
import ephys_atlas.data

pd.set_option('use_inf_as_na', True)
x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'cloud_x_std', 'cloud_y_std', 'cloud_z_std',
          'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
x_list += ['spike_count', 'peak_time_idx', 'peak_val', 'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val']
regions = BrainRegions()

## data loading section
config = ephys_atlas.data.get_config()
# this path contains channels.pqt, clusters.pqt and raw_ephys_features.pqt
path_features = Path(config['paths']['features']).joinpath(config['decoding']['tag'])
df_voltage, df_clusters, df_channels = ephys_atlas.data.load_tables(path_features)
df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()

aids_cosmos = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
aids_beryl = regions.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')
df_voltage['beryl_id'] = aids_beryl
df_voltage['cosmos_id'] = aids_cosmos

X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)

def decode(X, scaler, aids, classifier=None, save_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, aids, stratify=aids)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = classifier.fit(X_train, y_train)
    score_model = clf.score(X_test, y_test)
    y_null = aids_cosmos[np.random.randint(0, aids.size - 1, y_test.size)]
    score_null = clf.score(X_test, y_null)
    return clf, score_model, score_null

## %%
cclas, cs, csn = decode(X, scaler, aids_cosmos, classifier=RandomForestClassifier(verbose=True))
bclas, bs, bsn = decode(X, scaler, aids_beryl, classifier=RandomForestClassifier(verbose=True))
print(bs, bsn, cs, csn)

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
