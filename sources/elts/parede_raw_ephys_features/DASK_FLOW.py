
from pathlib import Path
import hashlib
import ephys_atlas.rawephys
from ephys_atlas.data import atlas_pids

from one.api import ONE
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")
# pids, alyx_pids = atlas_pids(one)
# pids.sort()


def destripe_ap(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)


def destripe_lf(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)


# python -m pip install dask distributed --upgrade
from distributed import Client
from distributed.diagnostics.progressbar import progress
client = Client()
client = Client('127.0.0.1:8786', process)

pids = atlas_pids()
for pid in pids:
    destripe_ap.submit(pid=pid)
    destripe_lf.submit(pid=pid)


futures = client.map(count_words, filenames)


# prefect server start

##
# pid = "99993a2b-588e-4c0c-bfec-e3dfb4f61534"
# destination = ROOT_PATH.joinpath(pid)
# ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)