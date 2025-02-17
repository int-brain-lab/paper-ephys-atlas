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
local_data_savebin = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data/bins')

# Load dataset
folder_seizure = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/seizure')
df_new = pd.read_parquet(folder_seizure.joinpath('col_5246af08.pqt')) # or 'col_5246af08-seizure.pqt'

# Select feature
# df_out = pd.DataFrame(index=df_new['channel'], columns=features)

for feature in features:

    # Load quantile for that feature
    namesave = f'{label}_{mapping}_{feature}'

    quantiles = np.load(local_data_savebin.joinpath(namesave + '.npy'))
    proba = pd.read_parquet(local_data_savebin.joinpath(namesave + '.pqt'))

    # For all channels at once, compute quantile idx for the given feature
    quantile_idx = np.searchsorted(quantiles, df_new[feature])
    # Localise the value for each channel using 1. the brain region 2. the quantile
    df_proba = proba.loc[df_new[mapping + '_id'], quantile_idx]
    # Take only diagonal terms
    df_new[feature + '_proba'] = pd.Series(np.diag(df_proba))

##
# Plot
from ephys_atlas.plots import figure_features_chspace
pid_ch_df = df_new.copy()
# Create numpy array of xy um (only 2D for plotting)
xy = df_new[['lateral_um', 'axial_um']].to_numpy()
# Plot
fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid='5246af08', mapping=mapping)
fig, axs = figure_features_chspace(pid_ch_df, [s + '_proba' for s in features], xy, pid='5246af08', mapping=mapping)
