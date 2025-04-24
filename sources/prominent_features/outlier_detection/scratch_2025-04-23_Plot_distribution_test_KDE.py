from ephys_atlas.plots import plot_histogram, plot_probe_rect2
from ephys_atlas.plots import select_series
import matplotlib.pyplot as plt
from ephys_atlas.data import load_voltage_features, download_tables
from ephys_atlas.features import voltage_features_set
from pathlib import Path
import seaborn
from one.api import ONE
##
import numpy as np
from sklearn.neighbors import KernelDensity

def detect_outliers_kde(train_data: np.ndarray, test_data: np.ndarray, kde=None):
    """
    Detects outliers in D-dimensional space using Kernel Density Estimation (KDE).

    Parameters:
    - train_data: (N, D) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M, D) numpy array, test dataset (points to evaluate for outlier probability).

    Returns:
    - outlier_probs: (M,) numpy array, probability of each test sample being an outlier.
    """
    # If kde is set, it is assumed to be alread trained
    if not kde:
        kde = KernelDensity()
        kde.fit(train_data)

    score_train = kde.score_samples(train_data)  # (N channel train,)
    score_test = kde.score_samples(test_data)  # (N channel test,)
    # We need to create a matrix
    # Put score train vertically, score test horizontally
    score_train = score_train[:, np.newaxis]
    score_test = score_test[np.newaxis, :]
    # We want the value of the KDE for the train samples to be higher than the test
    out = score_train >= score_test
    # This is the probability for the samples to be outliers
    return out.mean(axis=0)


##
one = ONE()
mapping = 'Beryl'
label = 'latest'
features = voltage_features_set()
local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data/23AprTest')

# -- Force download and load Ephys atlas DF
# download_tables(local_data_path, label=label, one=one, verify=False, overwrite=False, extended=False)
df_voltage, _, _, _ = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping) # df_clusters, df_channels, df_probes

# -- Check that features are in df columns
features = sorted(list(set(df_voltage.columns).intersection(set(features))))

features = ['rms_ap']

##
# Take PID into new DF and drop PID from df_voltage (baseline distribution)
pid = '0ce74616-abf8-47c2-86d9-f821cd25efd3'
idx = df_voltage[df_voltage['pids'] == pid].index

df_new = df_voltage.loc[idx].copy()
df_base = df_voltage.drop(idx).copy()


##
# Regions
regions = np.unique(df_new[mapping + '_id']).astype(int)
region = regions[1]

# Load data for that regions
df_region = select_series(df_base, features=features,
                          acronym=None, id=region, mapping=mapping)

# Get channel indices that are in region, keeping only feature values
df_new_compute = select_series(df_new, features=features,
                               acronym=None, id=region, mapping=mapping)

# Compute test/train sets
train_data = df_region.to_numpy()
test_data = df_new_compute.to_numpy()

score_out = detect_outliers_kde(train_data, test_data)
df_new_compute['score'] = score_out
##
# -- Plot example
feature = features[0]
fig, ax = plt.subplots()
series = select_series(df_voltage, feature, id=region)
plot_histogram(series, ax=ax, xlabel=feature, title=None)
# Tailored for alpha_mean : plot_histogram(series, ax=ax, xlabel=feature, title=None, bins=np.linspace(0,2000,100))
plt.show()

y = df_new_compute[feature].values
seaborn.scatterplot(data=df_new_compute, x=feature, y=2000, hue='score')
