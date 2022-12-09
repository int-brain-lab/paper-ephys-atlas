from pathlib import Path
from one.api import ONE
import shutil
from ephys_atlas.data import atlas_pids
import ephys_atlas.rawephys


VERSION = '1.3.0'
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

# 12, 33, 53
c = 0
IMIN = 0
for i, apid in enumerate(alyx_pids):
    if i < IMIN:
        continue
    pid = apid['id']
    eid = apid['session_info']['id']
    pname = apid['name']
    stub = [apid['session_info']['subject'], apid['session_info']['start_time'][:10], f"{apid['session_info']['number']:03d}"]
    destination = ROOT_PATH.joinpath('_'.join(([pid] + stub + [pname])))
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=True)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf', clobber=False)
    for flag in destination.glob('.01_destripe_1*'):
        flag.unlink()
    if destination.exists():
        destination.joinpath(f'.01_destripe_{VERSION}').touch()
