'''
Plot df voltage
'''
##
from pathlib import Path
import pandas as pd
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables
from ephys_atlas.plots import figure_features_chspace
from ephys_atlas.spatial_analysis import meta_spatial_derivative

br = BrainRegions()
label = '2023_W41'
brain_id = 'cosmos_id'

local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
local_result_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Fig3_Result')
local_fig_path = local_result_path.joinpath(brain_id)

df_voltage, df_clusters, df_channels, df_probes = load_tables(
    local_data_path.joinpath(label), verify=True)

df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
df_voltage['cosmos_id'] = br.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Cosmos')
df_voltage['beryl_id'] = br.remap(df_voltage['atlas_id'], source_map='Allen', target_map='Beryl')
# Do not remove void / root
##
# Prepare the dataframe for a single probe
pid = '0228bcfd-632e-49bd-acd4-c334cf9213e9'
pid_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

# Create numpy array of xy um (only 2D for plotting)
xy = pid_df[['lateral_um', 'axial_um']].to_numpy()

##
features = ['rms_lf', 'psd_delta', 'rms_ap']
feat_plt = features.copy()
for feature in features:
    feat_der = meta_spatial_derivative(pid_df, feature)
    new_name = f'{feature}_der'
    pid_df[new_name] = feat_der
    feat_plt.append(new_name)

##
# Select your features and plot
fig, axs = figure_features_chspace(pid_df, feat_plt, xy)

