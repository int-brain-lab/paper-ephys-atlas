from pathlib import Path
from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.rawephys

import torch

from ibllib.misc import check_nvidia_driver
from iblutil.util import get_logger

_logger = get_logger('ephys_atlas', level='INFO')

VERSION = '1.1.0'
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

check_nvidia_driver()
print(torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

c = 0
IMIN = 509
for i, apid in enumerate(alyx_pids):
    if i < IMIN:
        continue
    pid = apid['id']
    eid = apid['session_info']['id']
    pname = apid['name']
    stub = [apid['session_info']['subject'], apid['session_info']['start_time'][:10], f"{apid['session_info']['number']:03d}"]
    destination = ROOT_PATH.joinpath('_'.join(([pid] + stub + [pname])))
    if destination.is_dir():
        if destination.joinpath(f'.02_localisation_{VERSION}').exists():
            continue
        elif next(destination.glob(f'.01_destripe_1*'), None) is not None:
            print(i, f"COMPUTE {pid} --path {destination}")
            c += 1
            ephys_atlas.rawephys.localisation(destination)
            destination.joinpath(f'.02_localisation_{VERSION}').touch()
