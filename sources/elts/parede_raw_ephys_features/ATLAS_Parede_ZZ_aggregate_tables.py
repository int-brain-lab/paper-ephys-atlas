from pathlib import Path

import dask.dataframe as dd

from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

ROOT_PATH = Path("/mnt/s1/ephys-atlas")
one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, _ = atlas_pids(one)

flow = workflow.report(one=one, pids=pids)

# selects the pids that have no error in the flow
pids = flow.index[flow.applymap(lambda x: 'ERROR' not in x and x != '').all(axis=1)]

# load the tables, channels, clusters and raw features
files_channels_pqt = [p for p in ROOT_PATH.rglob('channels.pqt') if p.parts[-2] in pids]
channels = dd.read_parquet(files_channels_pqt).compute()

files_clusters_pqt = [p for p in ROOT_PATH.rglob('clusters.pqt') if p.parts[-2] in pids]
clusters = dd.read_parquet(files_clusters_pqt).compute()

files_raw_features = [p for p in ROOT_PATH.rglob('raw_ephys_features.pqt') if p.parts[-2] in pids]
raw_features = dd.read_parquet(files_raw_features).compute()

import pandas as pd
for f in files_raw_features:
    df = pd.read_parquet(f)
    print(df.index.names)

## %%
from ephys_atlas.workflow import compute_raw_features
pid = "b78b3c42-eee5-47c6-9717-743b78c0b721"

compute_raw_features(pid=pid)

import ephys_atlas.rawephys
import numpy as np
root_path = workflow.ROOT_PATH
ap_features = ephys_atlas.rawephys.compute_ap_features(pid, root_path=root_path)
lf_features = ephys_atlas.rawephys.compute_lf_features(pid, root_path=root_path)
spikes_features = ephys_atlas.rawephys.compute_spikes_features(pid, root_path=root_path)

spikes_features['channel'] = spikes_features['trace'].astype(np.int16)
channels_features = spikes_features.groupby('channel').agg(
    alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
    alpha_std=pd.NamedAgg(column="alpha", aggfunc="std"),
    spike_count=pd.NamedAgg(column="alpha", aggfunc="count"),
    cloud_x_std=pd.NamedAgg(column="x", aggfunc="std"),
    cloud_y_std=pd.NamedAgg(column="y", aggfunc="std"),
    cloud_z_std=pd.NamedAgg(column="z", aggfunc="std"),
    peak_trace_idx=pd.NamedAgg(column="peak_trace_idx", aggfunc="mean"),
    peak_time_idx=pd.NamedAgg(column="peak_time_idx", aggfunc="mean"),
    peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
    trough_time_idx=pd.NamedAgg(column="trough_time_idx", aggfunc="mean"),
    trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
)

channels_features = pd.merge(channels_features, ap_features, left_index=True, right_index=True)
channels_features = pd.merge(channels_features, lf_features, left_index=True, right_index=True)
# add the pid as the main index to prepare for concatenation
channels_features = pd.concat({pid: channels_features}, names=['pid'])
channels_features.to_parquet(root_path.joinpath(pid, 'raw_ephys_features.pqt'))