"""
Example script to debug a given task.
In this case we use the force_run flag to make sure the task runs and finds an eventual breakpoint.
"""

import ephys_atlas.workflow as workflow
from one.api import ONE

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pid = "2adc4f5d-bc7b-42a4-be76-f5df33d713d4"
workflow.compute_sorted_features(pid=pid, one=one, force_run=True)
