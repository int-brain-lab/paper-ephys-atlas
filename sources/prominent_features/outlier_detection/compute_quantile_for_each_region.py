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
local_data_savebin = local_data_path.joinpath('regions')
if not local_data_savebin.exists():
    local_data_savebin.mkdir()

features = voltage_features_set()
mapping = 'Allen'
label = 'latest'

df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)

regions = np.unique(df_voltage[mapping + '_id']).astype(int)

# Create dataframe with columns features / brain region / quantile (0-1) / value of the quantile division
nbin = 50

for region in regions:
    # df_save = pd.DataFrame(columns=['features', mapping + '_id', 'quantiles', 'values', 'n_samples'])
    # Note this could be made more efficient by pre-allocating the correct N rows

    # Get dataframe of single region
    df_region = df_voltage.loc[df_voltage[mapping + '_id'] == region]

    for count, feature in enumerate(features) :
        # Compute quantile boundaries
        df_compute = df_region[feature].copy()
        quantiles = np.linspace(0, 1, nbin+1)  # Does it make sense to save q=0 ? If not use [1:]
        values = df_compute.quantile(quantiles)
        n_samples = df_compute.shape[0]

        # Create dataframe
        df_create = pd.DataFrame({'features': feature,
                                   mapping + '_id': region,
                                  'quantiles': quantiles,
                                  'values': values,
                                  'n_samples': n_samples})

        # Concatenate dataframes
        if count == 0:
            df_save = df_create.copy()
        else:
            df_save = pd.concat([df_save, df_create])

    # Save dataframe
    namesave = f'df_quantile_{label}_{mapping}_{region}'
    df_save.to_parquet(local_data_savebin.joinpath(namesave + '.pqt'))
