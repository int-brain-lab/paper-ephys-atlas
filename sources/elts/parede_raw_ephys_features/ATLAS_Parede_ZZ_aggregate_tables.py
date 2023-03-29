from pathlib import Path

import dask.dataframe as dd

from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

ROOT_PATH = Path("/mnt/s1/ephys-atlas")
one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, _ = atlas_pids(one)

flow = workflow.report(one=one, pids=pids)






trials = dd.read_parquet(files_parquet)