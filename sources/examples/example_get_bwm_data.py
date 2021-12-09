from one.api import ONE
from brainbox.io.one import load_spike_sorting_fast
from ephys_atlas.data import bwm_pids

one = ONE()
pids, _ = bwm_pids(one)

for i, pid in enumerate(pids):
    eid, pname = one.pid2eid(pid)
    try:
        spikes, clusters, channels = load_spike_sorting_fast(eid=eid, probe=pname, one=one, nested=False)
        print(i, pid)
    except BaseException:
        pass
