from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ibllib.atlas import BrainRegions

regions = BrainRegions()

LOCAL_DATA_PATH = Path(r"C:\Users\jai\ibl\data")
OUT_PATH = Path(r"C:\Users\jai\ibl\data")

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
#df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
#df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
#df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))
#df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('raw_ephys_features.pqt'))

cl_features = [
    "amp_max", "amp_min", "amp_median", "amp_std_dB", "contamination",
    "contamination_alt", "drift", "missed_spikes_est", "noise_cutoff",
    "presence_ratio", "slidingRP_viol", "spike_count", "firing_rate",
    "atlas_id"
]
df_clusters = df_clusters[cl_features]
df_clusters.dropna(inplace=True)
cl_x_list = cl_features[:-1]
X = df_clusters[cl_x_list].values
y = df_clusters.atlas_id.values
aids_cosmos = regions.remap(y, source_map='Allen', target_map='Cosmos')
aids_beryl = regions.remap(y, source_map='Allen', target_map='Beryl')
scaler = StandardScaler()

def decode(X, scaler, y, classifier=None, save_path=None):
    if classifier is None:
        classifier = MLPClassifier(random_state=1, max_iter=300, verbose=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = classifier.fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    score_model = clf.score(X_test, y_test)
    y_null = y[np.random.randint(0, y.size - 1, y_test.size)]
    score_null = clf.score(X_test, y_null)
    return clf, score_model, score_null

scaler.fit(X)
cclf, csm, csn = decode(X, scaler, aids_cosmos, classifier=RandomForestClassifier(verbose=True))
forest_importances = pd.Series(cclf.feature_importances_, index=cl_x_list)
std = np.std([tree.feature_importances_ for tree in cclf.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# csm: 0.397, csn: 0.136
cclf, csm, csn = decode(X, scaler, aids_cosmos)
# csm: 0.372, csn: 0.140

# Note: error when training with Beryl regions: some regions only have 1 member: must have min of 2

fig, ax = plt.subplots()
bars = ax.bar(np.arange(4), (0.635, 0.543, 0.397, 0.372), edgecolor="black")
x_labels = ['cosmos_rf_V_decoding', 'cosmos_nn_V_decoding', 'cosmos_rf_cl_decoding', 'cosmos_nn_cl_decoding']
ax.set_xticks(np.arange(4))
ax.set_xticklabels(x_labels, rotation=-15)
bars[1].set_hatch("/")
bars[3].set_hatch("/")
bars[2].set_facecolor('r')
bars[3].set_facecolor('r')
ax.set_title("Cosmos Decoding Accuracy Across All Brain Regions")
fig.show()