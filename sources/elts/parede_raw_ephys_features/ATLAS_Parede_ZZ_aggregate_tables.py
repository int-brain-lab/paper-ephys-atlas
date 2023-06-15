from pathlib import Path
from datetime import date

import pandas as pd
import dask.dataframe as dd

from one.api import ONE

import ephys_atlas.data as data
import ephys_atlas.workflow as workflow

AGGREGATE_PATH = Path("/mnt/s1/aggregates/atlas")
ROOT_PATH = Path("/mnt/s0/ephys-atlas")
print(f'aws s3 sync "{ROOT_PATH}" s3://ibl-brain-wide-map-private/aggregates/atlas')

year_week = date.today().isocalendar()[:2]
OUT_PATH = Path(AGGREGATE_PATH).joinpath(f'{year_week[0]}_W{year_week[1]:02}')
OUT_PATH.mkdir(parents=True, exist_ok=True)

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, _ = data.atlas_pids(one)

flow = workflow.report(one=one, pids=pids)
# selects the pids that have no error in the flow
pids = flow.index[flow.applymap(lambda x: 'error' not in x and x != '').all(axis=1)]

#%% load the tables, channels, clusters and raw features
files_channels_pqt = [p for p in ROOT_PATH.rglob('channels.pqt') if p.parts[-2] in pids]
channels = pd.concat([pd.read_parquet(f) for f in files_channels_pqt])
channels.index.rename('channel', level=1, inplace=True)

files_clusters_pqt = [p for p in ROOT_PATH.rglob('clusters.pqt') if p.parts[-2] in pids]
clusters = pd.concat([pd.read_parquet(f) for f in files_clusters_pqt])
clusters.index.rename('cluster_id', level=1, inplace=True)

files_raw_features = [p for p in ROOT_PATH.rglob('raw_ephys_features.pqt') if p.parts[-2] in pids]
raw_features = dd.read_parquet(files_raw_features).compute()


#%% Prepare the insertions
channels, df_probes = data.compute_channels_micromanipulator_coordinates(channels)


#%%
channels.to_parquet(OUT_PATH.joinpath('channels.pqt'))
clusters.to_parquet(OUT_PATH.joinpath('clusters.pqt'))
raw_features.to_parquet(OUT_PATH.joinpath('raw_ephys_features.pqt'))
df_probes.to_parquet(OUT_PATH.joinpath('probes.pqt'))