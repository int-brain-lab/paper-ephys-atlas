"""
This module creates features and QC pictures for one insertion of the Atlas
"""
from pathlib import Path
import hashlib
import ephys_atlas.rawephys
from ephys_atlas.data import atlas_pids
from prefect.tasks import task_input_hash
import prefect.task_runners
import gc

from one.api import ONE
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
import shutil

for f in ROOT_PATH.rglob("raw.npy"):
    shutil.move(f, f.with_name('ap_raw.npy'))




LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")


def version_cache_key(context, parameters):
    """
    Concatenates the version into the re-run hash
    :param context:
    :param parameters:
    :return:
    """
    h = task_input_hash(context, parameters)
    return hashlib.md5((h + str(context.task.version)).encode('utf-8')).hexdigest()

task_kwargs = {
    "retries": 0,
    "cache_key_fn": version_cache_key,
}

@prefect.task(**task_kwargs, version="1.1", name="destripe ap {pid}")
def destripe_ap(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)


@prefect.task(**task_kwargs, version="1.1",name="destripe lf {pid}")
def destripe_lf(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf', clobber=False)


@prefect.flow(task_runner=prefect.task_runners.SequentialTaskRunner())
def compute_features():
    # creates a run with a name like "hello-marvin-on-Thursday"
    pids = [
        "99993a2b-588e-4c0c-bfec-e3dfb4f61534",
        "c4b5a9fa-10cb-4195-9c17-15b6a1f77f9a",
        "f93bfce4-e814-4ae3-9cdf-59f4dcdedf51",
    ]
    pids, alyx_pids = atlas_pids(one)
    pids.sort()
    for pid in pids:
        destripe_ap.submit(pid=pid)
        destripe_lf.submit(pid=pid)
        gc.collect()

compute_features()

# prefect server start

##
# pid = "99993a2b-588e-4c0c-bfec-e3dfb4f61534"
# destination = ROOT_PATH.joinpath(pid)
# ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)

# CREATE EXTENSION IF NOT EXISTS pgcrypto CASCADE;
# prefect orion start --host 0.0.0.0
# prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
# prefect config set PREFECT_ORION_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:Kraken45@localhost:5432/orion"

# on th elocal computer
#  ssh -NL localhost:4200:localhost:4200 ibladmin@broker-parede