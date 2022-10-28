from pathlib import Path
import time
import torch

from ibllib.misc import check_nvidia_driver
from iblutil.util import get_logger

_logger = get_logger('ephys_atlas', level='INFO')

VERSION = '1.0.0'
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

check_nvidia_driver()
print(torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"

ROOT_PATH = Path("/mnt/s1/ephys-atlas")


for ff in ROOT_PATH.rglob(".localise_me*"):
    print(f"python ./sources/ephys_atlas/rawephys.py localisation --path {ff.parent}")

# conda activate iblenv
# cd /home/ibladmin/Documents/PYTHON/int-brain-lab/paper-ephys-atlas
# python ./sources/elts/parede_raw_ephys_features/2022-07-26_ATLAS_Parede_02_localize.py > ~/parede_localise.sh
# source ~/parede_localise.sh
