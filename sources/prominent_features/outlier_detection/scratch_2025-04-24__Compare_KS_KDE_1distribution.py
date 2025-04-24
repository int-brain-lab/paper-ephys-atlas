from ephys_atlas.plots import plot_histogram, plot_probe_rect2
from ephys_atlas.plots import select_series
import matplotlib.pyplot as plt
from ephys_atlas.data import load_voltage_features, download_tables
from ephys_atlas.features import voltage_features_set
from pathlib import Path
import seaborn
from one.api import ONE
import numpy as np
from ephys_atlas.outliers import detect_outliers_kde, detect_outlier_kstest

##
mapping = 'Beryl'
label = 'latest'
features = voltage_features_set()
local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data/23AprTest')

# -- Force download and load Ephys atlas DF
# one = ONE()
# download_tables(local_data_path, label=label, one=one, verify=False, overwrite=False, extended=False)
df_voltage, _, _, _ = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping) # df_clusters, df_channels, df_probes

# -- Check that features are in df columns
features = sorted(list(set(df_voltage.columns).intersection(set(features))))

feature = 'depolarisation_slope' # features[1]

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
df_region = select_series(df_base, features=[feature],
                          acronym=None, id=region, mapping=mapping)

# Get channel indices that are in region, keeping only feature values
df_new_compute = select_series(df_new, features=[feature],
                               acronym=None, id=region, mapping=mapping)

# Compute test/train sets
train_data = df_region.to_numpy()
test_data = df_new_compute.to_numpy()

# --- KDE TEST ----
score_out = detect_outliers_kde(train_data, test_data)
df_new_compute[feature + '_kde_q'] = score_out
# Assign high and low values for picked threshold
p_thresh = 0.90
df_new_compute[feature + '_kde_extremes'] = 0
df_new_compute.loc[df_new_compute[feature + '_kde_q'] > p_thresh, feature + '_kde_extremes'] = 1


# --- KS TEST ----
score_out = detect_outlier_kstest(train_data, test_data)
df_new_compute[feature + '_ks_q'] = score_out
# Assign high and low values for picked threshold
p_thresh = 0.90
df_new_compute[feature + '_ks_extremes'] = 0
df_new_compute.loc[df_new_compute[feature + '_ks_q'] > p_thresh, feature + '_ks_extremes'] = 1

##
# -- Plot example
fig, ax = plt.subplots()
series = select_series(df_base, feature, id=region)
plot_histogram(series, ax=ax, xlabel=feature, title=None)
plt.show()

seaborn.scatterplot(data=df_new_compute, x=feature, y=1200, hue=f'{feature}_kde_extremes')
seaborn.scatterplot(data=df_new_compute, x=feature, y=1000, hue=f'{feature}_kde_q')

seaborn.scatterplot(data=df_new_compute, x=feature, y=2200, hue=f'{feature}_ks_extremes')
seaborn.scatterplot(data=df_new_compute, x=feature, y=2000, hue=f'{feature}_ks_q')
