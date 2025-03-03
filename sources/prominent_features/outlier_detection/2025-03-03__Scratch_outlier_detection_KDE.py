from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.data import load_voltage_features
import numpy as np
from sklearn.neighbors import KernelDensity
import pandas as pd
from pathlib import Path
from iblatlas.atlas import BrainRegions

## ========
# Definition of functions

def detect_outliers_kde(train_data: np.ndarray, test_data: np.ndarray):
    """
    Detects outliers in D-dimensional space using Kernel Density Estimation (KDE).

    Parameters:
    - train_data: (N, D) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M, D) numpy array, test dataset (points to evaluate for outlier probability).

    Returns:
    - outlier_probs: (M,) numpy array, probability of each test sample being an outlier.
    """
    kde = KernelDensity()
    kde.fit(train_data)
    mean_score = kde.score(train_data) / train_data.shape[0]
    out = 1 - np.exp(1-kde.score_samples(test_data) / mean_score)
    return out


def select_series_v2(df, features=None, acronym=None, id=None, mapping='Allen'):
    if features is None:  # Take the whole set
        features = voltage_features_set()
    if acronym is not None:
        series = df.loc[df[f'{mapping}_acronym'] == acronym, features]
    elif id is not None:
        series = df.loc[df[f'{mapping}_id'] == id, features]
    return series


# Put the below in: paper-ephys-atlas/sources/ephys_atlas/data.py
def prep_voltage_dataframe(df_voltage, mapping='Allen', regions=None):
    regions = BrainRegions() if regions is None else regions
    df_voltage.replace([np.inf, -np.inf], np.nan, inplace=True)
    if mapping != "Allen":
        df_voltage[mapping + "_id"] = regions.remap(
            df_voltage["Allen_id"], source_map="Allen", target_map=mapping
        )
        df_voltage[mapping + "_acronym"] = regions.id2acronym(
            df_voltage[mapping + "_id"]
        )

    return df_voltage

## ========
mapping = 'Beryl'
label = 'latest'
features = voltage_features_set()

# Path where distributions are saved
local_data_savebin = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data/regions')
local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')

# Load dataset
# === Ephys atlas DF
df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)
# === Seizure dataset
folder_seizure = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/seizure')
df_seiz = pd.read_parquet(folder_seizure.joinpath('col_5246af08.pqt')) # or 'col_5246af08-seizure.pqt'
df_new = prep_voltage_dataframe(df_seiz, mapping=mapping)

# # Pre-assign columns with 0 values
# for feature in features:
#     df_new[feature + '_q'] = 0

# Regions
regions = np.unique(df_new[mapping + '_id']).astype(int)

for count, region in enumerate(regions):

    # Load data for that regions
    df_region = select_series_v2(df_voltage, features=features,
                                 acronym=None, id=region, mapping=mapping)

    # Get channel indices that are in region
    idx_reg = np.where(df_new[mapping + '_id'] == region)
    df_new_compute = df_new.iloc[idx_reg].copy()

    for feature in features:
        # For all channels at once, test if outside the distribution for the given feature
        score_out = detect_outliers_kde(train_data=df_region[feature].values.reshape(-1, 1),
                                        test_data=df_new_compute[feature].values.reshape(-1, 1))

        # Save into new column
        df_new_compute[feature + '_q'] = score_out  # TODO check ordering of channels

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
