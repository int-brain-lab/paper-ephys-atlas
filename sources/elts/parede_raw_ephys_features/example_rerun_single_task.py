"""
This module regroups the ¨MAP" steps of the ephys atlas pipeline that compute features
-   destripe_ap, destripe_lf, compute_sorted Features
-   localize
-   compute_raw_features
"""

##
from pathlib import Path
from one.api import ONE

from ephys_atlas.data import atlas_pids, atlas_pids_autism
import ephys_atlas.workflow as workflow

RERUN_ERRORS = True

one = ONE(base_url="https://alyx.internationalbrainlab.org")

if True:
    ROOT_PATH = Path("/mnt/s0/ephys-atlas")
    pids, alyx_pids = atlas_pids(one)
else:
    ROOT_PATH = Path("/mnt/s0/ephys-atlas-autism")
    pids, alyx_pids = atlas_pids_autism(one)


report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()

# Runs the sorted features and raw features computations
pids_run = report.flow.get_pids_ready(
    "compute_sorted_features", include_errors=RERUN_ERRORS
)
for i, pid in enumerate(pids_run):
    print(i, len(pids_run), pid)
    workflow.compute_sorted_features(
        pid, one=one, data_path=ROOT_PATH, path_task=ROOT_PATH
    )


report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()


# from brainbox.io.one import SpikeSortingLoader
# pid = 'dada91ba-08a7-4e3c-824b-92f94506e8e5'
# ssl = SpikeSortingLoader(pid=pid, one=one)
