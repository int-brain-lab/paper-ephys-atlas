from pathlib import Path
from ephys_atlas.data import atlas_pids, atlas_pids_autism
import ephys_atlas.workflow as workflow

from one.api import ONE

one = ONE(base_url="https://alyx.internationalbrainlab.org")

if True:
    ROOT_PATH = Path("/mnt/s0/ephys-atlas")
    pids, alyx_pids = atlas_pids(one)
else:
    ROOT_PATH = Path("/mnt/s0/ephys-atlas-autism")
    pids, alyx_pids = atlas_pids_autism(one)


report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()
