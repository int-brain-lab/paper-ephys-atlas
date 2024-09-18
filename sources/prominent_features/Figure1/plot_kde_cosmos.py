# Plot KDE and similarity matrix for all regions, for a given feature
# Best display is on Cosmos
import matplotlib.pyplot as plt
from ephys_atlas.plots import plot_kde
from ephys_atlas.encoding import voltage_features_set, FEATURES_LIST
from ephys_atlas.data import load_tables, prepare_df_voltage
from pathlib import Path

label = '2024_W04'  # latest
mapping = 'Allen'
local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')
folder_file_save = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/Fig1/KDE')

##
df_voltage, df_clusters, df_channels, df_probes = load_tables(
    local_data_path.joinpath(label), verify=True)

df_voltage = prepare_df_voltage(df_voltage, df_channels)

features = voltage_features_set()
##
for id_feat, feature in enumerate(features):

    # Matrix
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches([11.88, 3.7])

    # KDE
    plot_kde(feature, df_voltage, ax=axs[0])
    plt.savefig(folder_file_save.joinpath(f'kde_sim__{feature}.svg'))
    plt.close()
