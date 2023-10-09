from pathlib import Path

import scipy.stats
import pandas as pd
import numpy as np

import ephys_atlas.data
import ephys_atlas.plots
from iblatlas.atlas import BrainRegions

DEFAULT_NQ = 25
br = BrainRegions()
local_path = Path("/Users/olivier/Documents/datadisk/paper-ephys-atlas/ephys-atlas-decoding/latest")
df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_tables(local_path, verify=True)
df = df_voltage.merge(df_channels, left_index=True, right_index=True)
df['atlas_id_beryl'] = br.remap(df['atlas_id'], source_map='Allen', target_map='Beryl')
# Index(['alpha_mean', 'alpha_std', 'spike_count', 'cloud_x_std', 'cloud_y_std',
#        'cloud_z_std', 'peak_trace_idx', 'peak_time_idx', 'peak_val',
#        'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val', 'rms_ap',
#        'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
#        'psd_gamma', 'x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um',
#        'lateral_um', 'histology', 'x_target', 'y_target', 'z_target',
#        'atlas_id_target'],
# df = df.groupby(['pid', 'atlas_id_beryl']).median(numeric_only=True)

def _agg_gini(x, quantiles):
    """
    Computes Gini coefficient for a pd.NamedAggregation function
    :param x:
    :param quantiles:
    :return:
    """
    feature_level = np.searchsorted(quantiles, x)
    qbins = np.bincount(feature_level, minlength=quantiles.size)
    return 1 - np.sum((qbins / np.sum(qbins)) ** 2)


def aggregations_dictionary(fseries, nq=DEFAULT_NQ):
    quantiles = fseries.quantile(np.linspace(0, 1, nq)[1:])
    aggregations = {
        f"{fseries.name}_median": pd.NamedAgg(column=fseries.name, aggfunc='median'),
        f"{fseries.name}_std": pd.NamedAgg(column=fseries.name, aggfunc='std'),
        f"{fseries.name}_gini": pd.NamedAgg(column=fseries.name, aggfunc=lambda x: _agg_gini(x, quantiles)),
    }
    return aggregations


## %%
features = ['rms_ap', 'rms_lf']
nq = DEFAULT_NQ
aggregations = {}
for f in features:
    aggregations.update(aggregations_dictionary(df[f]))
df_regions = df.groupby('atlas_id_beryl').agg(
    n_pids=pd.NamedAgg(column='atlas_id', aggfunc='count'),
    **aggregations
)

for feature in features:
    quantiles = df[feature].quantile(np.linspace(0, 1, nq)[1:])
    feature_level = np.searchsorted(quantiles, df[feature])
    df_regions['entropy'] = np.nan
    # feature_entropy = scipy.stats.entropy(np.bincount(feature_level, minlength=quantiles.size) / nq, base=2)
    df_regions[f'{feature}_info'] = np.nan
    for aid in df['atlas_id_beryl'].unique():
        isin = df['atlas_id_beryl'] == aid
        # compute frequencies for each quartile in and out of the region
        f = np.c_[  # f has shape (nq, 2)
            np.bincount(feature_level[isin], minlength=quantiles.size),
            np.bincount(feature_level[~isin], minlength=quantiles.size)]
        region_entropy = scipy.stats.entropy(np.sum(f, axis=0) / np.sum(f), base=2)
        # computes the remaining information as the weighted sum of each quartile entropy
        eq = np.nansum(- (p := f / np.sum(f, axis=1)[:, np.newaxis]) * np.log2(p), axis=1)
        remaining_information = np.sum(eq * np.sum(f, axis=1) / np.sum(f))
        df_regions.at[aid, f'{feature}_info_gain'] = 1 - remaining_information / region_entropy
        df_regions.at[aid, 'entropy'] = region_entropy
df_regions = df_regions.drop(index=0)

## %%
# ephys_atlas.plots.region_bars(atlas_id=df_regions.index.values, feature=df_regions['rms_ap_median'].values * 1e6, regions=br)br
##
# long wait per feature
# void creates an error
# type creates an error
# patch exisiting clunky
# add new feature doesn't work after url load
from server import FeatureUploader
up = FeatureUploader('olive')
acs = br.id2acronym(df_regions.index.values)
for feature in df_regions:
    if up.features_exist(feature):
        continue
    print(feature)
    up.create_features(feature, acs, df_regions[feature].values.astype(np.double), hemisphere='left')

url = up.get_buckets_url(['olive'])
print(url)