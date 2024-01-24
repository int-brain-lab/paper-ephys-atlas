'''
Plot df voltage
'''
##
from pathlib import Path
import pandas as pd
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_voltage_features
from ephys_atlas.plots import figure_features_chspace
from ephys_atlas.spatial_analysis import meta_spatial_derivative

br = BrainRegions()
label = '2023_W41'
mapping = 'Cosmos'

local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
local_result_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Fig3_Result')
local_fig_path = local_result_path.joinpath(mapping)

df_voltage, df_clusters, df_channels, df_probes = load_voltage_features(
    local_data_path.joinpath(label), regions=br, mapping=mapping)
# Do not remove void / root
##
# Prepare the dataframe for a single probe
# pid = '0228bcfd-632e-49bd-acd4-c334cf9213e9'
pid = '3d3d5a5e-df26-43ee-80b6-2d72d85668a5'

pid_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

# Create numpy array of xy um (only 2D for plotting)
xy = pid_df[['lateral_um', 'axial_um']].to_numpy()

##
# Compute spatial derivative metric for given set of features
features = ['rms_lf', 'psd_delta', 'rms_ap']
feat_plt = features.copy()
for feature in features:
    feat_der = meta_spatial_derivative(pid_df, feature)
    new_name = f'{feature}_der'
    pid_df[new_name] = feat_der
    feat_plt.append(new_name)

##
# Select your features and plot
fig, axs = figure_features_chspace(pid_df, feat_plt, xy, pid=pid, mapping=mapping)

## Note that because we drop nan from the df, channels can vary heavily in distance
import seaborn
seaborn.scatterplot(data=pid_df, x='lateral_um', y='axial_um')
