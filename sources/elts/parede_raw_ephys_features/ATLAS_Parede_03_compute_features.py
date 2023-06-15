import joblib

from one.api import ONE
from ephys_atlas.data import atlas_pids
import ephys_atlas.workflow as workflow

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

report = workflow.report(one=one)

pids = report.flow.get_pids_ready('compute_raw_features')
for i, pid in enumerate(pids):
    print(i, len(pids), pid)
    workflow.compute_raw_features(pid)

pids = report.flow.get_pids_ready('compute_sorted_features')
for i, pid in enumerate(pids):
    print(i, len(pids), pid)
    workflow.compute_sorted_features(pid, one=one)
