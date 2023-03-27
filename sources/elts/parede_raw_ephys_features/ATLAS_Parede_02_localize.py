from pathlib import Path
from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

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

flow = workflow.report()
pids = flow.index[flow['localise'] != f".localise_{workflow.TASKS['localise']['version']}"]
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.localise(pid)

