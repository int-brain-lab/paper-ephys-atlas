"""
This is the "Reduce" part of the ephys atlas pipeline
"""

from pathlib import Path
from datetime import date
import shutil

import numpy as np
import pandas as pd
import dask.dataframe as dd

from one.api import ONE
from iblutil.util import setup_logger

import ephys_atlas.data as data
import ephys_atlas.workflow as workflow

one = ONE(base_url="https://alyx.internationalbrainlab.org")
_logger = setup_logger(level="INFO")
AGGREGATE_PATH = Path("/mnt/s1/aggregates/atlas")

ROOT_PATH, label = (Path("/mnt/s0/ephys-atlas"), "")
pids, _ = data.atlas_pids(one)

# ROOT_PATH, label = (Path("/mnt/s0/ephys-atlas-autism"), "_autism")
# pids, _ = data.atlas_pids_autism(one)


print(
    f'aws --profile ibl s3 sync "{AGGREGATE_PATH}" s3://ibl-brain-wide-map-private/aggregates/atlas'
)

# the output path contains the tables, while the extended path contains large files that
# not all users may want to download
year_week = date.today().isocalendar()[:2]
OUT_PATH = Path(AGGREGATE_PATH).joinpath(f"{year_week[0]}_W{year_week[1]:02}{label}")
EXTENDED_PATH = Path(AGGREGATE_PATH).joinpath(
    f"{year_week[0]}_W{year_week[1]:02}{label}_extended"
)
# OUT_PATH = Path(AGGREGATE_PATH).joinpath(f'{year_week[0]}_W{4:02}{label}')
# EXTENDED_PATH = Path(AGGREGATE_PATH).joinpath(f'{year_week[0]}_W{4:02}{label}_extended')
OUT_PATH.mkdir(parents=True, exist_ok=True)
EXTENDED_PATH.mkdir(parents=True, exist_ok=True)

_logger.info("Checking current task status:")
flow = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
# selects the pids that have no error in the flow
pids = flow.index[flow.applymap(lambda x: "error" not in x and x != "").all(axis=1)]
flow.flow.print_report()

# %% load the tables, channels, clusters and raw features
# todo: should we use dask dataframes here ?
_logger.info("Concatenate channel tables")
files_channels_pqt = [p for p in ROOT_PATH.rglob("channels.pqt") if p.parts[-2] in pids]
channels = pd.concat([pd.read_parquet(f) for f in files_channels_pqt])
channels.index.rename("channel", level=1, inplace=True)

_logger.info("Concatenate clusters tables")
files_clusters_pqt = [p for p in ROOT_PATH.rglob("clusters.pqt") if p.parts[-2] in pids]
clusters = pd.concat([pd.read_parquet(f) for f in files_clusters_pqt])
clusters.index.rename("cluster_id", level=1, inplace=True)


def write_correlograms(file_name_correlograms):
    files_correlations = [
        f.with_name(file_name_correlograms) for f in files_clusters_pqt
    ]
    assert all([f.exists() for f in files_correlations])
    file_all_correlograms = EXTENDED_PATH.joinpath(
        f"clusters_{file_name_correlograms}"
    ).with_suffix(".bin")
    with open(file_all_correlograms, "wb+") as fid:
        for fc in files_correlations:
            arr = np.load(fc).T.copy()
            arr.tofile(fid)
    # test that we can memmap the file with the appropriate number of clusters
    np.memmap(
        file_all_correlograms, dtype="int32", shape=(clusters.shape[0], arr.shape[1])
    )
    return clusters.shape[0], arr.shape[1]


sz = write_correlograms(file_name_correlograms="correlograms_time_scale.npy")
_logger.info(f"correlograms time scales size {sz}")
_logger.info("correlograms refractory period")
write_correlograms(file_name_correlograms="correlograms_refractory_period.npy")

_logger.info("Aggregating ephys features tables")
files_raw_features = [
    p for p in ROOT_PATH.rglob("raw_ephys_features.pqt") if p.parts[-2] in pids
]
raw_features = dd.read_parquet(files_raw_features).compute()

_logger.info("Get micromanipulator coordinates")
channels, df_probes = data.compute_channels_micromanipulator_coordinates(
    channels, one=one
)

print(f"writing to {OUT_PATH}")
channels.to_parquet(OUT_PATH.joinpath("channels.pqt"))
clusters.to_parquet(OUT_PATH.joinpath("clusters.pqt"))
raw_features.to_parquet(OUT_PATH.joinpath("raw_ephys_features.pqt"))
df_probes.to_parquet(OUT_PATH.joinpath("probes.pqt"))


# %%
if label == "":  # this only applies to the main dataset
    shutil.rmtree(AGGREGATE_PATH.joinpath("latest"), ignore_errors=True)
    shutil.copytree(OUT_PATH, AGGREGATE_PATH.joinpath("latest"))
    AGGREGATE_PATH.joinpath("latest", f"{OUT_PATH.stem}.info").touch()

print(
    f'aws --profile ibl s3 sync "{AGGREGATE_PATH}" s3://ibl-brain-wide-map-private/aggregates/atlas'
)
