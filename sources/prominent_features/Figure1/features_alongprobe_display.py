##
from pathlib import Path
import pandas as pd
from ephys_atlas.plots import figure_features_chspace
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables, download_tables
from ephys_atlas.encoding import voltage_features_set
from one.api import ONE
import matplotlib.pyplot as plt

one = ONE()
br = BrainRegions()

pid = 'eebcaf65-7fa4-4118-869d-a084e84530e2'

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
# # Remove void / root
# df_voltage.drop(df_voltage[df_voltage['Allen_acronym'].isin(['void', 'root'])].index, inplace=True)

features = voltage_features_set()
##
# Get pid and plot
pid_ch_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()
xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()

# Select a pid and plot
fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid=pid, mapping=mapping)
plt.savefig(folder_file_save.joinpath(f'{pid}_features_{mapping}.svg'))
