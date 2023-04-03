from pathlib import Path

import dask.dataframe as dd

from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow
import pandas as pd
from datetime import date


year_week = date.today().isocalendar()[:2]
STAGING_PATH = Path('/mnt/s0/aggregates/atlas').joinpath(f'{year_week[0]}_W{year_week[1]:02}')
STAGING_PATH.mkdir(parents=True, exist_ok=True)

ROOT_PATH = Path("/mnt/s1/ephys-atlas")
one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, _ = atlas_pids(one)

flow = workflow.report(one=one, pids=pids)

# selects the pids that have no error in the flow
pids = flow.index[flow.applymap(lambda x: 'error' not in x and x != '').all(axis=1)]

# load the tables, channels, clusters and raw features
files_channels_pqt = [p for p in ROOT_PATH.rglob('channels.pqt') if p.parts[-2] in pids]
channels = pd.concat([pd.read_parquet(f) for f in files_channels_pqt])
channels.index.rename('channel', level=1, inplace=True)

files_clusters_pqt = [p for p in ROOT_PATH.rglob('clusters.pqt') if p.parts[-2] in pids]
clusters = pd.concat([pd.read_parquet(f) for f in files_clusters_pqt])
clusters.index.rename('cluster_id', level=1, inplace=True)

files_raw_features = [p for p in ROOT_PATH.rglob('raw_ephys_features.pqt') if p.parts[-2] in pids]
raw_features = dd.read_parquet(files_raw_features).compute()


channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))
clusters.to_parquet(STAGING_PATH.joinpath('clusters.pqt'))
raw_features.to_parquet(STAGING_PATH.joinpath('raw_ephys_features.pqt'))

print(f'aws s3 sync "/mnt/s0/aggregates/atlas" s3://ibl-brain-wide-map-private/aggregates/atlas')
