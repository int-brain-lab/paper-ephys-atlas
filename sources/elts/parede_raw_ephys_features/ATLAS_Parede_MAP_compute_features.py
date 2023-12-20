"""
This module regroups the ¨MAP" steps of the ephys atlas pipeline that compute features
-   destripe_ap, destripe_lf, compute_sorted Features
-   localize
-   compute_raw_features
"""

##
from pathlib import Path
import time
import torch

import joblib
from one.api import ONE

from ephys_atlas.data import atlas_pids, atlas_pids_autism
import ephys_atlas.workflow as workflow

assert torch.cuda.is_available(), "CUDA not available"
RERUN_ERRORS = False

one = ONE(base_url="https://alyx.internationalbrainlab.org")

if False:
    ROOT_PATH = Path("/mnt/s0/ephys-atlas")
    pids, alyx_pids = atlas_pids(one)
else:
    ROOT_PATH = Path("/mnt/s0/ephys-atlas-autism")
    pids, alyx_pids = atlas_pids_autism(one)


report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()


## Runs the destriping
pids_run = report.flow.get_pids_ready('destripe_ap', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids_run):
    print(i, len(pids_run))
    workflow.destripe_ap(pid=pid, one=one, data_path=ROOT_PATH, task_path=ROOT_PATH)


pids_run = report.flow.get_pids_ready('destripe_lf', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids_run):
    print(i, len(pids_run))
    workflow.destripe_lf(pid=pid, one=one, data_path=ROOT_PATH, task_path=ROOT_PATH)

report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()
## Runs the spike detection and localisationm this needs to run stand alone in a single rail

t = time.time()
pids_run = report.flow.get_pids_ready('localise', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids_run):
    print(i, len(pids_run))
    workflow.localise(pid, clobber=True, data_path=ROOT_PATH, path_task=ROOT_PATH)
print("localise", time.time() - t, len(pids_run))
report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()

## Runs the sorted features and raw features computations


pids_run = report.flow.get_pids_ready('compute_sorted_features', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids_run):
    print(i, len(pids_run), pid)
    workflow.compute_sorted_features(pid, one=one, data_path=ROOT_PATH, path_task=ROOT_PATH)

t = time.time()
pids_run = report.flow.get_pids_ready('compute_raw_features', include_errors=True)
joblib.Parallel(n_jobs=18)(joblib.delayed(workflow.compute_raw_features)(pid, data_path=ROOT_PATH, path_task=ROOT_PATH) for pid in pids)
print(time.time() - t, len(pids_run))


report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()


##
report = workflow.report(one=one, pids=pids, path_task=ROOT_PATH)
report.flow.print_report()