"""
This module regroups the ¨MAP" steps of the ephys atlas pipeline that compute features
-   destripe_ap, destripe_lf, compute_sorted Features
-   localize
-   compute_raw_features
"""

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




## Runs the destriping
report = workflow.report(one=one, pids=pids, task_path=ROOT_PATH)
report.flow.print_report()


pids = report.flow.get_pids_ready('destripe_ap', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.destripe_ap(pid=pid, one=one, task_path=ROOT_PATH)


pids = report.flow.get_pids_ready('destripe_lf', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.destripe_lf(pid=pid, one=one, task_path=ROOT_PATH)


## Runs the spike detection and localisationm this needs to run stand alone in a single rail
report = workflow.report(one=one, pids=pids, task_path=ROOT_PATH)
report.flow.print_report()

t = time.time()
pids = report.flow.get_pids_ready('localise', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.localise(pid, clobber=True, task_path=ROOT_PATH)
    print("destripe ap and lf bands", time.time() - t, len(pids))

## Runs the sorted features and raw features computations
report = workflow.report(one=one, pids=pids, task_path=ROOT_PATH)
report.flow.print_report()

pids = report.flow.get_pids_ready('compute_sorted_features', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids):
    print(i, len(pids), pid)
    workflow.compute_sorted_features(pid, one=one, task_path=ROOT_PATH)

t = time.time()
pids = report.flow.get_pids_ready('compute_raw_features', include_errors=RERUN_ERRORS)
joblib.Parallel(n_jobs=18)(joblib.delayed(workflow.compute_raw_features)(pid, task_path=ROOT_PATH) for pid in pids)
print(time.time() - t, len(pids))


##
report = workflow.report(one=one, pids=pids, task_path=ROOT_PATH)
report.flow.print_report()