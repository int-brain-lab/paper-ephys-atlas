import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables, download_tables
from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.plots import plot_kde, plot_similarity_matrix
from ephys_atlas.feature_information import feature_overall_entropy
from one.api import ONE

onen = ONE()
br = BrainRegions()

label = '2023_W34'  # label = '2023_W51_autism'
mapping = 'Cosmos'
local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
local_fig_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Anaysis_Entropy2024')
if not local_fig_path.exists():
    local_fig_path.mkdir()

force_download = False

local_data_path_clusters = local_data_path.joinpath(label).joinpath('clusters.pqt')
if not local_data_path_clusters.exists() or force_download:
    print('Downloading table')
    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
    df_voltage, df_clusters, df_channels, df_probes = download_tables(
        label=label, local_path=local_data_path, one=one)
else:
    df_voltage, df_clusters, df_channels, df_probes = load_tables(
        local_data_path.joinpath(label), verify=True)

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
df_voltage['pids'] = df_voltage.index.get_level_values(0)

df_voltage = df_voltage.rename(columns={"atlas_id": "Allen_id", "acronym": "Allen_acronym"})

df_voltage[mapping+'_id'] = br.remap(df_voltage['Allen_id'], source_map='Allen', target_map=mapping)
df_voltage[mapping+'_acronym'] = br.id2acronym(df_voltage[mapping+'_id'])
# Remove void / root
df_voltage.drop(df_voltage[df_voltage[mapping+'_acronym'].isin(['void', 'root'])].index, inplace=True)

features = voltage_features_set()

##
# Compute information gain per feature (overall) by pair of brain regions

dict_feat = dict()  # Keys will be the features, containing the dataframes

for feature in features:
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, 600)[1:])
    quantiles = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(df_voltage, values=feature, index=mapping + '_id', columns=quantiles, aggfunc='count')

    # Create a dataframe of Nregion x Nregion that will contain the entropy computed for a pair of region
    df_entropy = pd.DataFrame(index=counts.index, columns=counts.index)
    # Divide the counts into 2 regions
    for ireg1, reg1 in enumerate(counts.index):
        for ireg2, reg2 in enumerate(counts.index[ireg1+1:]):
            # increment the region index, so we do the comparison only once per pair
            ireg2_cmp = ireg1+ireg2+1
            # print(f'{ireg1}, {ireg2_cmp}')

            counts_reg = counts.iloc[[ireg1, ireg2_cmp], :]
            information_gain = feature_overall_entropy(counts_reg)

            # Save the result in both place in the DF
            df_entropy.at[reg1, reg2] = information_gain
            df_entropy.at[reg2, reg1] = information_gain

    dict_feat[feature] = df_entropy

df_multi = pd.concat(dict_feat.values(), axis=1, keys=dict_feat.keys())
# Replace Nans with zeros
df_multi.fillna(0, inplace=True)
##
# Display the matrix for a given feature
for feature in features:
    # Init figure
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches([11.88, 3.7])

    # KDE
    plot_kde(feature, df_voltage, ax=axs[0], brain_id=mapping + '_id')

    # Matrix
    plot_similarity_matrix(mat_plot=df_multi[feature].to_numpy(),
                           regions=df_multi[feature].index, ax=axs[1])


    # # SUM of information
    # df_info = df_multi[feature].sum().to_frame(name="Sum info")
    #
    # df_info[mapping + '_id'] = df_info.index.get_level_values(0)
    # df_info[mapping + '_acronym'] = br.id2acronym(df_info[mapping + '_id'])
    #
    # seaborn.barplot(data=df_info, x=mapping + '_acronym', y="Sum info", color='k', ax=axs[2])
    break
    # Save
    plt.savefig(local_fig_path.joinpath(f'entropy_sim__{label}_{mapping}__{feature}.png'))
    plt.close()
