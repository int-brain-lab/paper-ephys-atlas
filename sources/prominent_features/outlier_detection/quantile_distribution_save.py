'''
Compute distribution using bins, save quantile + bin count
'''
from pathlib import Path
import numpy as np
import pandas as pd
from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.data import load_voltage_features

local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')

# Create folder where to save ditributions
local_data_savebin = local_data_path.joinpath('bins')
if not local_data_savebin.exists():
    local_data_savebin.mkdir()

features = voltage_features_set()
mapping = 'Allen'
label = 'latest'

df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)

# Compute probability distribution using count in bins
nbin = 600
for feature in features:
    # Compute quantile boundaries
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, nbin)[1:])
    # Find in which quantile index is each sample
    quantiles_idx = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    # Note: counts give NAN in bins where there are 0 counts
    counts = pd.pivot_table(
        df_voltage,
        values=feature,
        index=mapping + "_id",
        columns=quantiles_idx,
        aggfunc="count",
    )

    # # Reformat index into brain region ID
    # counts = counts.rename_axis(mapping + "_id").reset_index()

    # Reformat quantiles series into numpy array
    quantiles = quantiles.to_numpy()

    # Save counts and quantiles
    namesave = f'{label}_{mapping}_{feature}'

    np.save(local_data_savebin.joinpath(namesave + '.npy'), quantiles)
    counts.to_parquet(local_data_savebin.joinpath(namesave + '.pqt'))
