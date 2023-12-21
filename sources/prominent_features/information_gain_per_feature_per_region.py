import numpy as np
from pathlib import Path
import pandas as pd

from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables, download_tables
from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.feature_information import feature_region_entropy
from one.api import ONE

onen = ONE()
br = BrainRegions()

label = '2023_W51_autism'
mapping = 'Beryl'
local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
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
# Compute region dataframe
df_regions = df_voltage.groupby(mapping + '_id').agg(
    n_channels=pd.NamedAgg(column=mapping + '_id', aggfunc='count'),
    n_pids=pd.NamedAgg(column='pids', aggfunc='nunique')
)
df_regions[mapping + '_id'] = df_regions.index.values
df_regions[mapping + '_acronym'] = br.id2acronym(df_regions.index.values, mapping=mapping)

df_regions = df_regions.sort_values('n_channels', ascending=False)

##
# Compute information gain per feature (per region)
# per region
# TODO review the code as the information gain is biased by the n_channels
for i_f, feature in enumerate(features):
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, 600)[1:])
    quantiles = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(df_voltage, values=feature, index=mapping + '_id', columns=quantiles, aggfunc='count')
    if i_f == 0:
        information_gain_region = pd.DataFrame(index=counts.index.values)
        information_gain_region[mapping + '_acronym'] = br.id2acronym(information_gain_region.index.values, mapping=mapping)
    information_gain_region[feature] = feature_region_entropy(counts)

information_gain_region = information_gain_region.merge(df_regions[['n_channels']], left_index=True, right_index=True)
# Place n_channel as second column
temp_cols = information_gain_region.columns.tolist()
new_cols = [temp_cols[0]] + [temp_cols[-1]] + temp_cols[1:-1]
information_gain_region = information_gain_region[new_cols]
# Sort
information_gain_region = information_gain_region.sort_values(features, ascending=False)
