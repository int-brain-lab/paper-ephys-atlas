from pathlib import Path
from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.rawephys

import torch

from ibllib.misc import check_nvidia_driver
from iblutil.util import setup_logger

_logger = setup_logger('ephys_atlas', level='INFO')

VERSION = '1.2.0'
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

check_nvidia_driver()
print(torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

# pids = [
#     '1a276285-8b0e-4cc9-9f0a-a3a002978724',
#     '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
#     '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
#     '5f7766ce-8e2e-410c-9195-6bf089fea4fd',
#     '6638cfb3-3831-4fc2-9327-194b76cf22e1',
#     '749cb2b7-e57e-4453-a794-f6230e4d0226',
#     'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
#     'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
#     'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
#     'dc7e9403-19f7-409f-9240-05ee57cb7aea',
#     'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
#     'eebcaf65-7fa4-4118-869d-a084e84530e2',
#     'fe380793-8035-414e-b000-09bfe5ece92a',
# ]

IMIN = 0
for i, pid in enumerate(pids):
    if i < IMIN:
        continue
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.localisation(destination, clobber=False)

    # if destination.is_dir():
    #     if destination.joinpath(f'.02_localisation_{VERSION}').exists():
    #         continue
    #     elif next(destination.glob(f'.01_destripe_1*'), None) is not None:
    #         print(i, f"COMPUTE {pid} --path {destination}")
    #         ephys_atlas.rawephys.localisation(destination)
    #         destination.joinpath(f'.02_localisation_{VERSION}').touch()


destination = Path("/mnt/s1/ephys-atlas/749cb2b7-e57e-4453-a794-f6230e4d0226")
ephys_atlas.rawephys.localisation(destination, clobber=True)

