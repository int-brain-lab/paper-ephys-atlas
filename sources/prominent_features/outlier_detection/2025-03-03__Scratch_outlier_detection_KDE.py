from ephys_atlas.encoding import voltage_features_set
from ephys_atlas.data import load_voltage_features
import numpy as np
import tqdm
from sklearn.neighbors import KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from iblatlas.atlas import BrainRegions
from scipy import stats

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

def detect_outlier_kstest(sample_values, cdf):
    pval = np.zeros(sample_values.shape)
    for count, sample in enumerate(sample_values):  # Test on each channel value independently
        ks_stat = stats.kstest(sample, cdf)
        pval[count] = ks_stat.pvalue
    return pval


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
mapping = 'Cosmos'
label = 'latest'
features = voltage_features_set() #[0:5]  # TODO remove, test on 2 features to begin with
TEST_TYPE = 'KSTest' #'KDE'

# Path where distributions are saved
local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')
folder_seizure = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/seizure')
# ROOT = Path(__file__).parent.parent.parent.parent
# local_data_path = ROOT / 'data'
# folder_seizure = ROOT / 'data/seizure'

# Load dataset
# === Ephys atlas DF
df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)
# === Seizure dataset
df_seiz = pd.read_parquet(folder_seizure.joinpath('col_5246af08-seizure.pqt')) # 'col_5246af08.pqt' or 'col_5246af08-seizure.pqt'
df_new = prep_voltage_dataframe(df_seiz, mapping=mapping)

# # Pre-assign columns with 0 values
# for feature in features:
#     df_new[feature + '_q'] = 0

# Regions
regions = np.unique(df_new[mapping + '_id']).astype(int)

for count, region in tqdm.tqdm(enumerate(regions), total=len(regions)):

    # Load data for that regions
    df_region = select_series_v2(df_voltage, features=features,
                                 acronym=None, id=region, mapping=mapping)

    # Get channel indices that are in region
    idx_reg = np.where(df_new[mapping + '_id'] == region)
    df_new_compute = df_new.iloc[idx_reg].copy()

    if TEST_TYPE == 'KSTest':

        for feature in features:
        # For all channels at once, test if outside the distribution for the given feature
            score_out =  detect_outlier_kstest(sample_values=df_new_compute[feature].values,
                                               cdf=df_region[feature].values)
            # Save into new column
            df_new_compute[feature + '_q'] = score_out  # TODO check ordering of channels

    elif TEST_TYPE == 'KDE':
        score_out = detect_outliers_kde(train_data=df_region[features].values,
                                        test_data=df_new_compute[features].values)
        # Save into new column
        df_new_compute['kde_q'] = score_out

    # Concatenate dataframes
    if count == 0:
        df_save = df_new_compute.copy()
    else:
        df_save = pd.concat([df_save, df_new_compute])

##
# Assign high and low values for picked threshold
if TEST_TYPE == 'KSTest':
    for feature in features:
        df_save[feature + '_extremes'] = 0
        df_save.loc[df_save[feature + '_q'] < 0.001, feature + '_extremes'] = 1
elif TEST_TYPE == 'KDE':
    df_save.loc[df_save['kde_q'] > 0.9, 'kde_extremes'] = 1

##
# Plot
from ephys_atlas.plots import figure_features_chspace
pid_ch_df = df_save.copy()
# Create numpy array of xy um (only 2D for plotting)
xy = pid_ch_df[['lateral_um', 'axial_um']].to_numpy()
# Plot
if TEST_TYPE == 'KSTest':
    fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid='5246af08', mapping=mapping)
    fig, axs = figure_features_chspace(pid_ch_df, [s + '_q' for s in features], xy, pid='5246af08', mapping=mapping)
    fig, axs = figure_features_chspace(pid_ch_df, [s + '_extremes' for s in features], xy, pid='5246af08', mapping=mapping)
elif TEST_TYPE == 'KDE':
    fig, axs = figure_features_chspace(pid_ch_df, features, xy, pid='5246af08', mapping=mapping)
    fig, axs = figure_features_chspace(pid_ch_df, ['kde_q'], xy, pid='5246af08', mapping=mapping)
    fig, axs = figure_features_chspace(pid_ch_df, ['kde_extremes'], xy, pid='5246af08', mapping=mapping)


plt.show()
