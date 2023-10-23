'''
Plot of KDEs and similarity matrix
'''
# from ephys_atlas.prominent_features import ... TODO
from pathlib import Path
import matplotlib.pyplot as plt
from iblatlas.atlas import BrainRegions
import numpy as np

from ephys_atlas.plots import plot_kde, plot_similarity_matrix
from ephys_atlas.data import prepare_mat_plot, load_tables, prepare_df_voltage

br = BrainRegions()
label = '2023_W41'
brain_id = 'cosmos_id'

local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
local_result_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Fig3_Result')
local_fig_path = local_result_path.joinpath(brain_id)
#
# results_log, regions, features = load_ks_result(
#     local_result_path, test_todo='ks-test', brain_id=brain_id, label=label)
features = ['alpha_mean', 'alpha_std'] # to debug

df_voltage, df_clusters, df_channels, df_probes = load_tables(
    local_data_path.joinpath(label), verify=True)

df_voltage = prepare_df_voltage(df_voltage, df_channels)

##
# Plot KDEs for two regions only
# regions_id = br.acronym2id(['AON', 'DG'])
# regions_id = np.array([1065, 512])
regions_id = br.acronym2id(['HY', 'CB'])
# Single figure with subplot per feature
n_sub = int(np.ceil(np.sqrt(len(features))))
fig, axs = plt.subplots(n_sub, n_sub)
fig.set_size_inches([11.88, 3.7])
axs = axs.flatten()

for id_feat, feature in enumerate(features):

    # KDE
    plot_kde(feature, df_voltage, ax=axs[id_feat], regions_id=regions_id)
    plt.savefig(local_fig_path.joinpath(f'kde__{regions_id}.png'))
    plt.close()


##
# Plot KDE and similarity matrix for all regions, for a given feature
# Best display is on Cosmos
arr_results = results_log
for id_feat, feature in enumerate(features):
    mat_plot = prepare_mat_plot(arr_results, id_feat)

    # Matrix
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches([11.88, 3.7])

    plot_similarity_matrix(mat_plot, regions, ax=axs[1])

    # KDE
    plot_kde(feature, df_voltage, ax=axs[0])
    plt.savefig(local_fig_path.joinpath(f'kde_sim__{feature}.png'))
    plt.close()
