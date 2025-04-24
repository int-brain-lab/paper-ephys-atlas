from ephys_atlas.plots import plot_histogram, plot_probe_rect2
from ephys_atlas.plots import select_series
import matplotlib.pyplot as plt
from ephys_atlas.data import load_voltage_features, download_tables
from ephys_atlas.features import voltage_features_set
from pathlib import Path
import pandas as pd
import tqdm
from one.api import ONE
from ephys_atlas.outliers import detect_outliers_kde
import numpy as np

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
# features = ['alpha_mean', 'rms_lf', 'depolarisation_slope']

##
# Take PID into new DF and drop PID from df_voltage (baseline distribution)
pid = '091392a5-73f6-40f3-8552-fa917cf96deb'
idx = df_voltage[df_voltage['pids'] == pid].index

df_new = df_voltage.loc[idx].copy()
df_base = df_voltage.drop(idx).copy()

##
# Regions
regions = np.unique(df_new[mapping + '_id']).astype(int)

for count, region in tqdm.tqdm(enumerate(regions), total=len(regions)):

    # Get channel indices that are in region, but keeping all info besides features
    idx_reg = np.where(df_new[mapping + '_id'] == region)
    df_new_compute = df_new.iloc[idx_reg].copy()

    for feature in features:
        # Load data for that regions
        df_train = select_series(df_base, features=[feature],
                                 acronym=None, id=region, mapping=mapping)

        # Get channel indices that are in region, keeping only feature values
        df_test = select_series(df_new, features=[feature],
                                acronym=None, id=region, mapping=mapping)

        # For all channels at once, test if outside the distribution for the given features
        train_data = df_train.to_numpy()
        test_data = df_test.to_numpy()
        score_out = detect_outliers_kde(train_data, test_data)
        # Save into new column
        df_new_compute[feature + '_q'] = score_out  # TODO check ordering of channels

    # Concatenate dataframes
    if count == 0:
        df_save = df_new_compute.copy()
    else:
        df_save = pd.concat([df_save, df_new_compute])

##
# Assign high and low values for picked threshold
p_thresh = 0.90
for feature in features:
    df_save[feature + '_extremes'] = 0
    df_save.loc[df_save[feature + '_q'] > p_thresh, feature + '_extremes'] = 1

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

##
# Plot distribution
import seaborn
feature = features[2]
fig, ax = plt.subplots()
series = select_series(df_base, feature, id=region)
plot_histogram(series, ax=ax, xlabel=feature, title=None)
# Tailored for alpha_mean : plot_histogram(series, ax=ax, xlabel=feature, title=None, bins=np.linspace(0,2000,100))
plt.show()

idx_reg = np.where(df_save[mapping + '_id'] == region)
df_plot = df_save.iloc[idx_reg].copy()
seaborn.scatterplot(data=df_plot, x=feature, y=1200, hue=f'{feature}_extremes')
seaborn.scatterplot(data=df_plot, x=feature, y=1000, hue=f'{feature}_q')