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



gb_regions = df_voltage.groupby('atlas_id_beryl')
percentiles = np.arange(10) / 10
quantile_funcs = [(p, lambda x: x.quantile(p)) for p in percentiles]
df_regions = gb_regions.agg({
    'rms_ap': ('median', 'var', *quantile_funcs),
    'rms_lf': ('median', 'var', *quantile_funcs),
    'psd_gamma': 'median',
    'psd_delta': 'median',
    'psd_alpha': 'median',
    'psd_beta': 'median',
    'psd_theta': 'median',
    'spike_rate': 'median',
    'bad_channel': 'sum',
    'acronym': 'first',
    'x': 'count',
})

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=2)


from iblutil.numerical import ismember
_, df_voltage['ir'] = ismember(df_voltage['atlas_id_beryl'].values, ba.regions.id)


sel_columns = ['psd_delta', 'ir',
       'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'rms_ap', 'rms_lf']
df = df_voltage[sel_columns].select_dtypes(['number']).dropna()
# TODO filter out bad channels

pca_result = pca.fit_transform(df.drop(columns=['ir']))
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
import matplotlib.pyplot as plt

plt.scatter( pca_result[:,0],  pca_result[:,1], 5, df['ir'])


plt.figure()
import seaborn as sns
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="ir",
    data=df,
    size=.1,
    alpha=0.3,
    palette=sns.color_palette("magma", as_cmap=True)
)


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.drop(columns=['ir']))

df['tsne-one'] = tsne_results[:,0]
df['tsne-two'] = tsne_results[:,1]

plt.figure()
import seaborn as sns
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="ir",
    data=df,
    size=.1,
    alpha=0.3,
    palette=sns.color_palette("magma", as_cmap=True)
)
