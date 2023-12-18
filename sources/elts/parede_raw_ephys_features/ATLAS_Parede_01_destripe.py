from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

RERUN_ERRORS = False
one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

report = workflow.report(one=one)

pids = report.flow.get_pids_ready('destripe_ap', include_errors=RERUN_ERRORS)
len(pids)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.destripe_ap(pid, one)


pids = report.flow.get_pids_ready('destripe_lf', include_errors=RERUN_ERRORS)
for i, pid in enumerate(pids):
    print(i, len(pids))
    workflow.destripe_lf(pid, one)
