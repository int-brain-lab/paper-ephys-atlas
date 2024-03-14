'''
Plot df voltage
'''
##
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_voltage_features
from ephys_atlas.plots import figure_features_chspace
from ephys_atlas.encoding import voltage_features_set

br = BrainRegions()
label = '2023_W51'
mapping = 'Allen'

local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
save_folder = Path(f'/Users/gaelle/Documents/Work/EphysAtlas/Fig3_Result/{mapping}')

if not save_folder.parent.exists():
    save_folder.parent.mkdir()
if not save_folder.exists():
    save_folder.mkdir()

df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)
# Do not remove void / root
##
# Prepare the dataframe for a single probe
# pid = '0ee04753-3039-4209-bed8-5c60e38fe5da'
# pid = '0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a'
# pid ='5810514e-2a86-4a34-b7bd-1e4b0b601295' # TODO mark as critical
# pid = 'f362c84f-8d9a-4d5b-8439-055ae936fdff'
pid = '0ee04753-3039-4209-bed8-5c60e38fe5da'
pid_ch_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

# Create numpy array of xy um (only 2D for plotting)
xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()

##
# Select your features and plot
plot_all = True
if plot_all:
    features = voltage_features_set()
else:  # Select your own features to plot
    features = ['rms_lf', 'psd_delta', 'rms_ap']
#
fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid=pid, mapping=mapping)
# fig.tight_layout()
plt.savefig(save_folder.joinpath(f'{pid}.pdf'))
