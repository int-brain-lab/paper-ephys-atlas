'''
Example plot of features onto channel layout in 2D space
'''
from pathlib import Path
from ephys_atlas.plots import figure_features_chspace
from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.data import load_voltage_features

local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')

features = voltage_features_set()
mapping = 'Allen'
label = 'latest'

df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)

# Plot
pids = df_voltage.index.get_level_values(0).unique()
for i_pid, pid in enumerate(pids[0:1]):  # Take the first PID for show
    print(i_pid)
    # Prepare the dataframe for a single probe
    pid_ch_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()
    pid_ch_df = pid_ch_df.droplevel(level=0)  # Drop pid
    # Create numpy array of xy um (only 2D for plotting)
    xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()
    # Plot
    fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid='5246af08', mapping=mapping)
