from pathlib import Path

from one.api import ONE
import ephys_atlas.data

LABEL = '2024_W04'
LOCAL_DATA_PATH = Path('/home/ibladmin/scratch/')
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(
    label=LABEL, local_path=LOCAL_DATA_PATH, one=one, extended=True)

corr_ts = ephys_atlas.data.read_correlogram(LOCAL_DATA_PATH.joinpath(
    LABEL, 'clusters_correlograms_time_scale.bin'), df_clusters.shape[0])
corr_rf = ephys_atlas.data.read_correlogram(LOCAL_DATA_PATH.joinpath(
    LABEL, 'clusters_correlograms_refractory_period.bin'), df_clusters.shape[0])


# %% optionally look at the data
from easyqc.gui import viewseis
viewseis(corr_ts[100_000:110_000, :], 0.001)
viewseis(corr_rf[100_000:110_000, :], 1 / 30000)
