from pathlib import Path
import numpy as np

from one.api import ONE
from ibllib.atlas import AllenAtlas

from ibllib.ephys.neuropixel import trace_header
from iblutil.numerical import ismember2d
from ephys_atlas.data import bwm_pids
from brainbox.io.one import SpikeSortingLoader

one = ONE()
ba = AllenAtlas()


import pandas as pd

STAGING_PATH = Path('/datadisk/FlatIron/tables/bwm')
STAGING_PATH.mkdir(exist_ok=True, parents=True)

excludes = [
    'd8c7d3f2-f8e7-451d-bca1-7800f4ef52ed',  # key error in loading histology from json
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',  # same same
    'c2184312-2421-492b-bbee-e8c8e982e49e',  # same same
    '58b271d5-f728-4de8-b2ae-51908931247c',  # same same
    'f86e9571-63ff-4116-9c40-aa44d57d2da9',  # 404 not found
]

one = ONE()
pids, _ = bwm_pids(one, tracing=True)

# init dataframes
df_probes = pd.DataFrame(dict(eid='', pname='', spike_sorter='', histology=''), index=pids)
ldf_channels = []
ldf_clusters = []
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
    print(i, pid)
    ss = SpikeSortingLoader(pid, one=one, atlas=ba)
    spikes, clusters, channels = ss.load_spike_sorting()
    df_probes['spike_sorter'][i] = ss.collection
    df_probes['histology'][i] = ss.histology
    if not spikes:
        no_spike_sorting.append(pid)
        continue
    metrics = clusters.pop('metrics')
    df_ch = pd.DataFrame(channels)
    df_ch['pid'] = pid
    ldf_channels.append(df_ch)
    df_clu = pd.merge(metrics, pd.DataFrame(clusters), left_index=True, right_index=True)
    df_clu['pid'] = pid
    ldf_clusters.append(df_clu)

df_channels = pd.concat(ldf_channels, ignore_index=True)
df_clusters = pd.concat(ldf_clusters, ignore_index=True)

# convert the channels dataframe to a multi-index dataframe
h = trace_header(version=1)
_, chind = ismember2d(df_channels.loc[:, ['lateral_um', 'axial_um']].to_numpy(), np.c_[h['x'], h['y']])
df_channels['raw_ind'] = chind
df_channels = df_channels.set_index(['pid', 'raw_ind'])

# saves the 3 dataframes
df_channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))
df_clusters.to_parquet(STAGING_PATH.joinpath('clusters.pqt'))
df_probes.to_parquet(STAGING_PATH.joinpath('probes.pqt'))
