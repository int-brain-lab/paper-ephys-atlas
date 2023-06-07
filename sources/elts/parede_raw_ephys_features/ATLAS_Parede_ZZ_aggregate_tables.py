from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import dask.dataframe as dd

from iblutil.numerical import ismember
from one.api import ONE

from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow


ROOT_PATH = Path("/mnt/s0/aggregates/atlas")
print(f'aws s3 sync "{ROOT_PATH}" s3://ibl-brain-wide-map-private/aggregates/atlas')

year_week = date.today().isocalendar()[:2]
STAGING_PATH = Path(ROOT_PATH).joinpath(f'{year_week[0]}_W{year_week[1]:02}')
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


## Prepare the insertions
from ibllib.atlas import Insertion
from ibllib.atlas import NeedlesAtlas, AllenAtlas
from ibllib.pipes.histology import interpolate_along_track
needles = NeedlesAtlas()
allen = AllenAtlas()

pids = channels.index.levels[0]
trajs = one.alyx.rest('trajectories', 'list', provenance='Micro-Manipulator')

mapping = dict(
    pid='probe_insertion', pname='probe_name', x_target='x', y_target='y', z_target='z',
    depth_target='depth', theta_target='theta', phi_target='phi', roll_target='roll'
)

tt = [{k: t[v] for k, v in mapping.items()} for t in trajs]
df_planned = pd.DataFrame(tt).rename(columns={'probe_insertion': 'pid'}).set_index('pid')
df_planned['eid'] = [t['session']['id'] for t in trajs]

df_probes = channels.groupby('pid').agg(histology=pd.NamedAgg(column='histology', aggfunc='first'))
df_probes = pd.merge(df_probes, df_planned, how='left', on='pid')

iprobes, iplan = ismember(pids, df_planned.index)
imiss = np.where(~iprobes)[0]

for pid, rec in df_probes.iterrows():
    drec = rec.to_dict()
    ins = Insertion.from_dict({v: drec[k] for k, v in mapping.items() if 'target' in k})
    txyz = np.flipud(ins.xyz)
    txyz = allen.bc.i2xyz(needles.bc.xyz2i(txyz / 1e6, round=False, mode="clip")) * 1e6
    # we interploate the channels from the deepest point up. The neuropixel y coordinate is from the bottom of the probe
    xyz_mm = interpolate_along_track(txyz, channels.loc[pid, 'axial_um'].to_numpy() / 1e6)
    aid_mm = needles.get_labels(xyz=xyz_mm, mode='clip')

    channels.loc[pid, 'x_target'] = xyz_mm[:, 0]
    channels.loc[pid, 'y_target'] = xyz_mm[:, 1]
    channels.loc[pid, 'z_target'] = xyz_mm[:, 2]
    channels.loc[pid, 'atlas_id_target'] = aid_mm


##
channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))
clusters.to_parquet(STAGING_PATH.joinpath('clusters.pqt'))
raw_features.to_parquet(STAGING_PATH.joinpath('raw_ephys_features.pqt'))
df_probes.to_parquet(STAGING_PATH.joinpath('probes.pqt'))