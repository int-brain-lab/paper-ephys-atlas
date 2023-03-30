from pathlib import Path
from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

flow = workflow.report(one=one)

pids = workflow.get_pids_for_task(task_name='compute_raw_features', flow=flow, n_workers=2, worker_id=1)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.compute_raw_features(pid)


pids = workflow.get_pids_for_task(task_name='compute_sorted_features', flow=flow, n_workers=2, worker_id=1)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.compute_sorted_features(pid, one=one)