import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables
from ephys_atlas.encoding import voltage_features_set, FEATURES_LIST
from ephys_atlas.plots import color_map_feature
from ephys_atlas.feature_information import feature_overall_entropy

br = BrainRegions()

label = '2023_W41'
mapping = 'beryl_id'
local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')

df_voltage, df_clusters, df_channels, df_probes = load_tables(
    local_data_path.joinpath(label), verify=True)

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
df_voltage['cosmos_id'] = br.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
df_voltage['beryl_id'] = br.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')
# Do not remove void / root
df_voltage['pids'] = df_voltage.index.get_level_values(0)

features = voltage_features_set()

##
# Compute information gain per feature (overall)
information_gain = dict()
for feature in features:
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, 600)[1:])
    quantiles = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(df_voltage, values=feature, index=mapping, columns=quantiles, aggfunc='count')
    information_gain[feature] = feature_overall_entropy(counts)

information_gain = pd.DataFrame(information_gain, index=['information_gain']).T.sort_values('information_gain', ascending=False).reset_index()
##
# Plot histogram of information gain per feature, color coded by feature type

# Add new column "color"
color_set = color_map_feature()
information_gain['color'] = 0  # init new column
for feature_i, color_i in zip(FEATURES_LIST, color_set):
    feat_list = voltage_features_set(feature_i)
    indx = information_gain['index'].isin(feat_list)
    information_gain['color'][indx] = color_i

# Plot
fig, ax = plt.subplots()
sns.barplot(information_gain, y='information_gain', x='index', palette=information_gain['color'])
plt.xticks(rotation=90)
fig.tight_layout()
