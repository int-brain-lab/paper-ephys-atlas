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
    # # Compute bin boundaries using histogram
    overall_counts, divisions = np.histogram(df_voltage[feature], bins=nbin)

    # Find in which division index is each sample
    division_idx = np.searchsorted(divisions, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(
        df_voltage,
        values=feature,
        index=mapping + "_id",
        columns=division_idx,
        aggfunc="count",
    )
    # Note: counts give Nan in bins where there are 0 counts. Replace Nan with 0
    counts = counts.fillna(0)

    # Columns that did not have a hit in division_idx are generated and filled with 0 :
    diff_col = set(np.linspace(0, nbin-1, nbin)) - set(np.unique(division_idx))
    diff_col = np.fromiter(diff_col,int)
    df_0 = pd.DataFrame(0, index=counts.index, columns=diff_col)
    # Merge
    counts = counts.join(df_0)

    # Divide counts by sum over rows to get probabilities
    # NOTE THIS IS WRONG IF NOT USING SAME SIZE BIN
    sum_rows = counts.sum(axis=1)
    proba = counts.div(sum_rows, axis=0)
    # Test to check sum over row == 1
    np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)
    # Save counts and quantiles
    namesave = f'{label}_{mapping}_{feature}'

    np.save(local_data_savebin.joinpath(namesave + '.npy'), divisions)
    proba.to_parquet(local_data_savebin.joinpath(namesave + '.pqt'))


# Debug plot
if False:
    import matplotlib.pyplot as plt
    plt.stairs(overall_counts, divisions)
    plt.stairs(counts.iloc[0, :], divisions)
    plt.show()
