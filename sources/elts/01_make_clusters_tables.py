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

STAGING_PATH = Path('/mnt/s0/aggregates/bwm')

excludes = []
errorkey = []
error404 = []
one = ONE(base_url='https://alyx.internationalbrainlab.org')
pids, _ = bwm_pids(one, tracing=True)

## %%
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
df_depths = pd.concat(ldf_depths)

# convert the channels dataframe to a multi-index dataframe
h = trace_header(version=1)
_, chind = ismember2d(df_channels.loc[:, ['lateral_um', 'axial_um']].to_numpy(), np.c_[h['x'], h['y']])
df_channels['raw_ind'] = chind
df_channels = df_channels.set_index(['pid', 'raw_ind'])

# convert the depths dataframe to a multi-index dataframe
df_depths['depths'] = df_depths.index.values
df_depths = df_depths.set_index(['pid', 'depths'])

# saves the 3 dataframes
STAGING_PATH.mkdir(exist_ok=True, parents=True)
df_channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))
df_clusters.to_parquet(STAGING_PATH.joinpath('clusters.pqt'))
df_probes.to_parquet(STAGING_PATH.joinpath('probes.pqt'))
df_depths.to_parquet(STAGING_PATH.joinpath('depths.pqt'))



print(f'aws s3 sync "{STAGING_PATH}" s3://ibl-brain-wide-map-private/aggregates/bwm')
print(errorkey)
print(error404)

# Run August 22nd 2022
# In [35]: error404
# Out[35]:
# ['b2ea68e2-c732-4d17-8166-1a8595fff225',
#  '577e4741-4b15-4e91-b81b-61304a09bfb5',
#  '9b3ad89a-177f-4242-9a96-2fd98721e47f',
#  '367ea4c4-d1b7-47a6-9f18-5d0df9a3f4de',
#  '4279e354-a6b8-4eff-8245-7c8723b07834',
#  '507b20cf-4ab0-4f55-9a81-88b02839d127',
#  'f2a098e7-a67e-4125-92d8-36fc6b606c45',
#  'f967a527-257f-404a-871d-b91575dca3b4',
#  'ffb1b072-2de7-44a4-8115-5799b9866382',
#  'de5d704c-4a5b-4fb0-b4af-71614c510a8b',
#  '2201eb05-aebe-4bda-905e-b5cc1baf3840',
#  '39180bcb-13e5-46f9-89e1-8ea2cba22105',
#  '04db6b9e-a80c-4507-a98e-ad76294ac444',
#  '531423f6-d36d-472b-8234-c8f7b8293f79']
