"""
This module creates features and QC pictures for one insertion of the Atlas
"""
from pathlib import Path
import prefect
import hashlib
import ephys_atlas.rawephys
from prefect.tasks import task_input_hash

from one.api import ONE
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")
# pids, alyx_pids = atlas_pids(one)
# pids.sort()

from prefect import get_client


async with get_client() as client:
    # set a concurrency limit of 10 on the 'small_instance' tag

    client.delete_concurrency_limit_by_tag(tag='parede')
    limit_id = await client.create_concurrency_limit(tag="parede", concurrency_limit=4)
    limit_id = await client.create_concurrency_limit(tag="gpu", concurrency_limit=1)


def version_cache_key(context, parameters):
    """
    Concatenates the version into the re-run hash
    :param context:
    :param parameters:
    :return:
    """
    h = task_input_hash(context, parameters)
    return hashlib.md5((h + str(context.task.version)).encode('utf-8')).hexdigest()


@prefect.task(retries=0, cache_key_fn=version_cache_key, version="1.1", tags=['parede'])
def destripe_ap(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)


@prefect.task(retries=0, cache_key_fn=version_cache_key, version="1.1", tags=['parede'])
def destripe_lf(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)


@prefect.flow
def compute_features():
    # creates a run with a name like "hello-marvin-on-Thursday"
    pids = [
        "99993a2b-588e-4c0c-bfec-e3dfb4f61534",
        "c4b5a9fa-10cb-4195-9c17-15b6a1f77f9a",
        "f93bfce4-e814-4ae3-9cdf-59f4dcdedf51",
    ]
    for pid in pids:
        destripe_ap.submit(pid=pid)
        destripe_lf.submit(pid=pid)

compute_features()

# prefect server start

##
# pid = "99993a2b-588e-4c0c-bfec-e3dfb4f61534"
# destination = ROOT_PATH.joinpath(pid)
# ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)