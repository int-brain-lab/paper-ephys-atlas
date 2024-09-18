##
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables, download_tables
from ephys_atlas.encoding import voltage_features_set, FEATURES_LIST
from ephys_atlas.plots import color_map_feature
from ephys_atlas.feature_information import feature_overall_entropy
from one.api import ONE

one = ONE()
br = BrainRegions()

label = '2024_W04'  # latest
mapping = 'Allen'
local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')
folder_file_save = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/Fig1')
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
df_voltage = df_voltage.rename(columns={"atlas_id": "Allen_id", "acronym": "Allen_acronym"})

df_voltage['Cosmos_id'] = br.remap(df_voltage['Allen_id'], source_map='Allen', target_map='Cosmos')
df_voltage['Beryl_id'] = br.remap(df_voltage['Allen_id'], source_map='Allen', target_map='Beryl')
df_voltage['pids'] = df_voltage.index.get_level_values(0)
# Remove void / root
df_voltage.drop(df_voltage[df_voltage['Allen_acronym'].isin(['void', 'root'])].index, inplace=True)

features = voltage_features_set()

##
# Compute information gain per feature (overall)
information_gain = dict()
for feature in features:
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, 600)[1:])
    quantiles = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(df_voltage, values=feature, index=mapping + '_id', columns=quantiles, aggfunc='count')
    information_gain[feature] = feature_overall_entropy(counts)

information_gain = pd.DataFrame(information_gain, index=['information_gain']).T.sort_values(
    'information_gain', ascending=False).reset_index()
##
# Plot histogram of information gain per feature, color coded by feature type

# Add new column "color"

color_set = color_map_feature(feature_list=FEATURES_LIST, default=False,
                              return_dict=True)

information_gain['feat_type'] = ''  # init new column
for feature_i in color_set.keys():
    feat_list = voltage_features_set(feature_i)
    indx = information_gain['index'].isin(feat_list)
    information_gain.loc[indx, 'feat_type'] = feature_i

# Plot
fig, ax = plt.subplots()
sns.barplot(information_gain, y='information_gain', x='index',
            hue='feat_type', palette=color_set)
plt.xticks(rotation=90)
fig.tight_layout()
# plt.show()

# Save figure
plt.savefig(folder_file_save.joinpath(f"info_gain_overall_{label}_{mapping}.svg"))
plt.savefig(folder_file_save.joinpath(f"info_gain_overall_{label}_{mapping}.pdf"),
            format="pdf", bbox_inches="tight")
plt.close()
