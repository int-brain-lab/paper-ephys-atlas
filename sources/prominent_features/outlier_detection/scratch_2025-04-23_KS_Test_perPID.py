from ephys_atlas.plots import plot_histogram, plot_probe_rect2
from ephys_atlas.plots import select_series
import matplotlib.pyplot as plt
from ephys_atlas.data import load_voltage_features, download_tables
from ephys_atlas.features import voltage_features_set
from pathlib import Path
import pandas as pd
import tqdm
from one.api import ONE
##
from scipy import stats
import numpy as np

def detect_outlier_kstest(sample_values, cdf):
    pval = np.zeros(sample_values.shape)
    for count, sample in enumerate(sample_values):  # Test on each channel value independently
        ks_stat = stats.kstest(sample, cdf)
        pval[count] = ks_stat.pvalue
    return pval
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

##
# Take PID into new DF and drop PID from df_voltage (baseline distribution)
pid = '0ce74616-abf8-47c2-86d9-f821cd25efd3'
idx = df_voltage[df_voltage['pids'] == pid].index

df_new = df_voltage.loc[idx].copy()
df_base = df_voltage.drop(idx).copy()
##
# -- Plot example
feature = 'alpha_mean'
fig, ax = plt.subplots()
series = select_series(df_voltage, feature, acronym='void')
plot_histogram(series, ax=ax, xlabel=feature, title=None, bins=np.linspace(0,2000,100))
plt.show()

y = df_new[feature].values
plt.scatter(y, 2000 * np.ones(y.shape))



##
# Regions
regions = np.unique(df_new[mapping + '_id']).astype(int)

for count, region in tqdm.tqdm(enumerate(regions), total=len(regions)):

    # Load data for that regions
    df_region = select_series(df_base, features=features,
                              acronym=None, id=region, mapping=mapping)

    # Get channel indices that are in region
    idx_reg = np.where(df_new[mapping + '_id'] == region)
    df_new_compute = df_new.iloc[idx_reg].copy()


    for feature in features:
    # For all channels at once, test if outside the distribution for the given feature
        score_out =  detect_outlier_kstest(sample_values=df_new_compute[feature].values,
                                           cdf=df_region[feature].values)
        # Save into new column
        df_new_compute[feature + '_q'] = score_out  # TODO check ordering of channels


    # Concatenate dataframes
    if count == 0:
        df_save = df_new_compute.copy()
    else:
        df_save = pd.concat([df_save, df_new_compute])

##
# Assign high and low values for picked threshold
p_thresh = 0.01
for feature in features:
    df_save[feature + '_extremes'] = 0
    df_save.loc[df_save[feature + '_q'] < p_thresh, feature + '_extremes'] = 1

##
# Plot
from ephys_atlas.plots import figure_features_chspace
pid_ch_df = df_save.copy()
# Create numpy array of xy um (only 2D for plotting)
xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()
# Plot
fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid=pid, mapping=mapping, plot_rect=plot_probe_rect2)
fig, axs = figure_features_chspace(pid_ch_df, [s + '_q' for s in features], xy, pid=pid, mapping=mapping, plot_rect=plot_probe_rect2)
fig, axs = figure_features_chspace(pid_ch_df, [s + '_extremes' for s in features], xy, pid=pid, mapping=mapping, plot_rect=plot_probe_rect2)

plt.show()
