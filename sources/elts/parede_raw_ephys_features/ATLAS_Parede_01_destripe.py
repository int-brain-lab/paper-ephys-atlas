from pathlib import Path
from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

flow = workflow.report(one=one)

# re-runs all old and error tasks
pids = flow.index[flow['destripe_ap'] != f".destripe_ap_{workflow.TASKS['destripe_ap']['version']}"]
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.destripe_ap(pid, one)


pids = flow.index[flow['destripe_lf'] != f".destripe_lf_{workflow.TASKS['destripe_lf']['version']}"]
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.destripe_lf(pid, one)
