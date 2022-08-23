from pathlib import Path
import numpy as np
import pandas as pd
import urllib.error

from one.api import ONE
from ibllib.atlas import AllenAtlas

from neuropixel import trace_header
from iblutil.numerical import ismember2d
from ephys_atlas.data import bwm_pids
from brainbox.io.one import SpikeSortingLoader
from iblutil.util import get_logger

logger = get_logger('ibl')
one = ONE()
ba = AllenAtlas()


STAGING_PATH = Path('/datadisk/FlatIron/tables/atlas')
STAGING_PATH = Path('/mnt/s0/Data/tables/atlas')

excludes = []
errorkey = []
error404 = []
one = ONE(base_url='https://alyx.internationalbrainlab.org')
pids, _ = bwm_pids(one, tracing=True)

# init dataframes
df_probes = pd.DataFrame(dict(eid='', pname='', spike_sorter='', histology=''), index=pids)
ldf_channels = []
ldf_clusters = []
ldf_depths = []
no_spike_sorting = []
IMIN = 0

for i, pid in enumerate(pids):
    if i < IMIN:
        continue
    if pid in excludes:
        continue
    eid, pname = one.pid2eid(pid)
    df_probes['eid'][i] = eid
    df_probes['pname'][i] = pname

    # spikes, clusters, channels = load_spike_sorting_fast(eid=eid, probe=pname, one=one, nested=False)
    logger.info(f"{i}, {pid}")
    ss = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    try:
        spikes, clusters, channels = ss.load_spike_sorting()
    except urllib.error.HTTPError:
        error404.append(pid)
        logger.error(f"{pid} error 404")
        continue
    except KeyError:
        errorkey.append(pid)
        logger.error(f"{pid} key error")
        continue
    clusters = ss.merge_clusters(spikes, clusters, channels)
    df_probes['spike_sorter'][i] = ss.collection
    df_probes['histology'][i] = ss.histology
    if not spikes:
        no_spike_sorting.append(pid)
        continue
    df_ch = pd.DataFrame(channels)
    df_ch['pid'] = pid
    ldf_channels.append(df_ch)
    df_clu = pd.DataFrame(clusters)
    df_clu['pid'] = pid
    ldf_clusters.append(df_clu)
    # aggregate spike features per depth
    df_spikes = pd.DataFrame(spikes)
    df_spikes.dropna(axis=0, how='any', inplace=True)
    df_spikes['rdepths'] = (np.round(df_spikes['depths'] / 20) * 20).astype(np.int32)
    df_spikes['amps'] = df_spikes['amps'] * 1e6
    df_depths = df_spikes.groupby('rdepths').agg(
        amps=pd.NamedAgg(column="amps", aggfunc="median"),
        amps_std=pd.NamedAgg(column="amps", aggfunc="std"),
        cell_count=pd.NamedAgg(column="clusters", aggfunc="nunique"),
        spike_rate=pd.NamedAgg(column="amps", aggfunc="count"),
    )
    df_depths['pid'] = pid
    df_depths['spike_rate'] = df_depths['spike_rate'] / (np.max(spikes['times']) - np.min(spikes['times']))
    ldf_depths.append(df_depths)

df_channels = pd.concat(ldf_channels, ignore_index=True)
df_clusters = pd.concat(ldf_clusters, ignore_index=True)
df_depths = pd.concat(ldf_depths, ignore_index=True)

# convert the channels dataframe to a multi-index dataframe
h = trace_header(version=1)
_, chind = ismember2d(df_channels.loc[:, ['lateral_um', 'axial_um']].to_numpy(), np.c_[h['x'], h['y']])
df_channels['raw_ind'] = chind
df_channels = df_channels.set_index(['pid', 'raw_ind'])

# convert the depths dataframe to a multi-index dataframe
df_depths = df_depths.set_index(['pid', 'r_depths'])

# saves the 3 dataframes
STAGING_PATH.mkdir(exist_ok=True, parents=True)
df_channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))
df_clusters.to_parquet(STAGING_PATH.joinpath('clusters.pqt'))
df_probes.to_parquet(STAGING_PATH.joinpath('probes.pqt'))


print(f'aws s3 sync "{STAGING_PATH}" s3://ibl-brain-wide-map-private/data/tables/atlas')
print(errorkey)
print(error404)
