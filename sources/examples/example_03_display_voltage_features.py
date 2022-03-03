from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from ibllib.atlas import AllenAtlas


ba = AllenAtlas()


STAGING_PATH = Path('/datadisk/FlatIron/tables/bwm')

df_clusters = pd.read_parquet(STAGING_PATH.joinpath('clusters.pqt'))
df_probes = pd.read_parquet(STAGING_PATH.joinpath('probes.pqt'))
# df_channels = pd.read_parquet(STAGING_PATH.joinpath('channels.pqt'))

df_voltage = pd.read_parquet(STAGING_PATH.joinpath('channels_voltage_features.pqt'))
df_voltage['atlas_id_beryl'] = ba.regions.remap(df_voltage['atlas_id'].to_numpy(), target_map='Beryl')

ca1ids = ba.regions.descendants(ba.regions.acronym2id('CA1'))['id']



# BANDS = {'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12], 'beta': [15, 30], 'gamma': [30, 90]}


import seaborn as sns

sns.set_theme()
sns.set_context("paper")
key = 'psd_gamma'
val = df_voltage[key].values
val = val[~np.isnan(val)]
pds, x = np.histogram(val, density=True, bins=200)


fig, ax = plt.subplots()
ax.plot(x[0] + np.cumsum(np.diff(x)), pds, label='whole brain')

val = df_voltage[key].values
val = val[np.isin(df_voltage['atlas_id'].values, ca1ids)]
val = val[~np.isnan(val)]
pds, x = np.histogram(val, density=True, bins=50)

ax.plot(x[0] + np.cumsum(np.diff(x)), pds, label='CA1')
ax.set(xlabel='PSD: dB relative to v/sqrt(Hz)', ylabel='Probability density', title=key)
ax.legend()








gb_regions = df_voltage.groupby('atlas_id_beryl')


df_regions = gb_regions.agg({
    'rms_ap': ('median', 'var'),
    'rms_lf': ('median', 'var'),
    'psd_gamma': 'median',
    'acronym': 'first',
    'x': 'count',
})

# Plot PSD on coronal slice
# From https://int-brain-lab.github.io/iblenv/notebooks_external/atlas_plotting_scalar_on_slice.html
from ibllib.atlas.plots import plot_scalar_on_slice

acro_plot = df_regions['acronym'].values.tolist()
data_plot = df_regions['psd_gamma'].values.tolist()
acro_plot_flat = np.array([item for sublist in acro_plot for item in sublist])
data_plot_flat = np.array([item for sublist in data_plot for item in sublist])
fig, ax = plot_scalar_on_slice(acro_plot_flat, data_plot_flat, coord=-1800, slice='coronal', mapping='Allen', hemisphere='both',
                               background='boundary', cmap='viridis', brain_atlas=ba)
# TODO add colorbar ?

from iblutil.numerical import ismember

_, ich = ismember(df_regions.index, ba.regions.id[ba.regions.mappings['Beryl']])
image_map_beryl = ba.regions.mappings['Beryl'][ba.label.flat[image_map]]  # this indexing maps directly
exists_in_map, ind_image_map_channels = ismember(image_map_beryl, ich)

picture = np.zeros_like(image_map)
picture[exists_in_map] = df_regions['x'].to_numpy().flatten()[ind_image_map_channels]


ba.regions.id[ba.regions.mappings['Beryl'][ba.label.flat[image_map]]]



fig, ax = plt.subplots(4, 1, figsize=(16, 5))
ax[0].imshow(ba._label2rgb(ba.label.flat[image_map]), origin='upper')
ax[1].imshow(ba._label2rgb(ba.regions.mappings['Beryl'][ba.label.flat[image_map]]), origin='upper')
ax[2].imshow(ba._label2rgb(ba.regions.mappings['Cosmos'][ba.label.flat[image_map]]), origin='upper')
ax[2].imshow(picture, origin='upper')