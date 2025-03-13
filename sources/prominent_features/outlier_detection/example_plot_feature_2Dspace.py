'''
Example plot of features onto channel layout in 2D space
'''
from pathlib import Path
from one.api import ONE

import matplotlib.pyplot as plt

from ephys_atlas.plots import figure_features_chspace, plot_probe_rect, plot_probe_rect2
from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.data import load_voltage_features, download_tables

local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')
force_download = True
features = voltage_features_set()
mapping = 'Allen'
label = 'latest'

if force_download:
    print("Downloading table")
    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="local")
    _, _, _, _ = download_tables(label=label, local_path=local_data_path, one=one,
                                 overwrite=True)

df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)

##
# Plot
pids = df_voltage.index.get_level_values(0).unique()
n = 1
cmap = "Spectral" # RdYlBu, PuOr, managua
for i_pid, pid in enumerate(pids[0:n]):  # Take the first PID for show
    print(i_pid)
    # Prepare the dataframe for a single probe
    pid_ch_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()
    pid_ch_df = pid_ch_df.droplevel(level=0)  # Drop pid
    # Create numpy array of xy um (only 2D for plotting)
    xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()
    # Plot
    fig, axs = figure_features_chspace(
        pid_ch_df, features, xy, pid='5246af08', mapping=mapping, plot_rect=plot_probe_rect, cmap=cmap)
    fig, axs = figure_features_chspace(
        pid_ch_df, features, xy, pid='5246af08', mapping=mapping, plot_rect=plot_probe_rect2, cmap=cmap)
plt.show()

##
# Plot specific pid
pid = "5246af08-0730-40f7-83de-29b5d62b9b6d"
cmap = "Spectral"
# Prepare the dataframe for a single probe
pid_ch_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()
pid_ch_df = pid_ch_df.droplevel(level=0)  # Drop pid
# Create numpy array of xy um (only 2D for plotting)
xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()
# Plot
fig, axs = figure_features_chspace(
    pid_ch_df, features, xy, pid='5246af08', mapping=mapping, plot_rect=plot_probe_rect2, cmap=cmap)
plt.show()
