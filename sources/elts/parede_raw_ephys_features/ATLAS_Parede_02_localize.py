from pathlib import Path

import torch

from one.api import ONE
from ibllib.misc import check_nvidia_driver
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

ROOT_PATH = Path("/mnt/s1/ephys-atlas")

check_nvidia_driver()
assert torch.cuda.is_available(), "CUDA not available"
print(torch.version.cuda)

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

# re-runs all old and error tasks
report = workflow.report(one=one)

pids = report.flow.get_pids_ready('localise')
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.localise(pid)
