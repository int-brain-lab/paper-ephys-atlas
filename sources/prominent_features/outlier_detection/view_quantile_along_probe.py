'''
Load a new recording - feature values, brain region labels are given per channel
Per channel/feature, compute its distribution quantile and associated probability
'''

from pathlib import Path
import numpy as np
import pandas as pd
from ephys_atlas.encoding import voltage_features_set

features = voltage_features_set()
mapping = 'Allen'
label = 'latest'

# Path where distributions are saved
local_data_savebin = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data/regions')

# Load dataset
folder_seizure = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/seizure')
df_new = pd.read_parquet(folder_seizure.joinpath('col_5246af08.pqt')) # or 'col_5246af08-seizure.pqt'

# # Pre-assign columns with 0 values
# for feature in features:
#     df_new[feature + '_q'] = 0

# Regions
regions = np.unique(df_new[mapping + '_id']).astype(int)

for count, region in enumerate(regions):

    # Load quantile for that regions
    namesave = f'df_quantile_{label}_{mapping}_{region}'
    df_region = pd.read_parquet(local_data_savebin.joinpath(namesave + '.pqt'))

    # Get channel indices that are in region
    idx_reg = np.where(df_new[mapping + '_id'] == region)
    df_new_compute = df_new.iloc[idx_reg].copy()

    for feature in features:
        df_compute = df_region.loc[df_region['features'] == feature]

        # For all channels at once, compute quantile idx for the given feature
        quantile_idx = np.searchsorted(df_compute['values'], df_new_compute[feature])
        # Replace by max quantile
        idx_replace = np.where(quantile_idx == df_compute.shape[0])[0]
        quantile_idx[idx_replace] = df_compute.shape[0] - 1

        # Get the quantile for each channel
        df_new_compute[feature + '_q'] = df_compute['quantiles'].iloc[quantile_idx].values

    # Concatenate dataframes
    if count == 0:
        df_save = df_new_compute.copy()
    else:
        df_save = pd.concat([df_save, df_new_compute])

##
# Assign high and low values for picked quantile threshold
for feature in features:
    df_save[feature + '_extremes'] = 0
    df_save.loc[df_save[feature + '_q'] > 0.9, feature + '_extremes'] = 1
    df_save.loc[df_save[feature + '_q'] < 0.1, feature + '_extremes'] = -1

##
# Plot
from ephys_atlas.plots import figure_features_chspace
pid_ch_df = df_save.copy()
# Create numpy array of xy um (only 2D for plotting)
xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()
# Plot
fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid='5246af08', mapping=mapping)
fig, axs = figure_features_chspace(pid_ch_df, [s + '_q' for s in features], xy, pid='5246af08', mapping=mapping)
fig, axs = figure_features_chspace(pid_ch_df, [s + '_extremes' for s in features], xy, pid='5246af08', mapping=mapping)
