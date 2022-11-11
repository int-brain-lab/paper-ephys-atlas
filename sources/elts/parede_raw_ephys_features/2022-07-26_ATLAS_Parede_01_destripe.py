from pathlib import Path
from one.api import ONE
import shutil
from ephys_atlas.data import atlas_pids
import ephys_atlas.rawephys


VERSION = '1.1.0'
ROOT_PATH = Path("/mnt/s1/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, alyx_pids = atlas_pids(one)

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
    if destination.is_dir():
        if destination.joinpath(f'.01_destripe_{VERSION}').exists():
            continue
        elif destination.joinpath(f'.01_destripe_1.0.0').exists():
            print(i, f"UPDATE {pid} --path {destination}")
            ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf')
            destination.joinpath('.01_destripe_1.0.0').rename(f'.01_destripe_{VERSION}')
            continue
    print(i, f"COMPUTE {pid} --path {destination}")
    c += 1
    shutil.rmtree(destination, ignore_errors=True)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf')
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap')
    destination.joinpath(f'.01_destripe_{VERSION}').touch()
