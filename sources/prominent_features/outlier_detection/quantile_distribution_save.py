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

    counts = pd.pivot_table(
        df_voltage,
        values=feature,
        index=mapping + "_id",
        columns=quantiles_idx,
        aggfunc="count",
    )
    # Note: counts give NAN in bins where there are 0 counts. Replace Nan with 0
    counts = counts.fillna(0)
    # Divide counts by sum over rows to get probabilities
    sum_rows = counts.sum(axis=1)
    proba = counts.div(sum_rows, axis=0)
    # Test to check sum over row == 1
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)

    # Reformat quantiles series into numpy array
    quantiles = quantiles.to_numpy()

    # Save counts and quantiles
    namesave = f'{label}_{mapping}_{feature}'

    np.save(local_data_savebin.joinpath(namesave + '.npy'), quantiles)
    proba.to_parquet(local_data_savebin.joinpath(namesave + '.pqt'))
