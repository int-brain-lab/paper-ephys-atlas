from pathlib import Path
import shutil
import numpy as np

from one.api import ONE
import spikeglx
from neurodsp.voltage import decompress_destripe_cbin
from viewephys.gui import viewephys

one = ONE()
ROOT_PATH_NEW = Path("/media/olivier/Seagate Expansion Drive/hackathon")
ROOT_PATH = Path("/datadisk/FlatIron")

# '/media/olivier/Seagate Expansion Drive/hackathon/zadorlab/Subjects/CSH_ZAD_025/2020-08-04/002/raw_ephys_data/probe01/destripe/_spikeglx_ephysData_g0_t0.imec1.lf.bin
pid = "64d04585-67e7-4320-baad-8d4589fd18f7"  # 280ee768-f7b8-4c6c-9ea0-48ca75d6b6f3	probe01	64d04585-67e7-4320-baad-8d4589fd18f7	zadorlab/Subjects/CSH_ZAD_025/2020-08-04/002 CA1
pid = "31d8dfb1-71fd-4c53-9229-7cd48bee07e4"  # 493170a6-fd94-4ee4-884f-cc018c17eeb9	probe01	31d8dfb1-71fd-4c53-9229-7cd48bee07e4	hoferlab/Subjects/SWC_061/2020-11-23/001	MOs

# stub = "zadorlab/Subjects/CSH_ZAD_025/2020-08-04/002"
# print(f'rsync -av --progress "/media/olivier/Seagate Expansion Drive/hackathon/{stub}/" /datadisk/FlatIron/{stub}')

eid, pname = one.pid2eid(pid)
session_path = one.eid2path(eid)
session_path = ROOT_PATH_NEW.joinpath(session_path.relative_to(ROOT_PATH))
proc_path = session_path.joinpath('raw_ephys_data', pname, 'destripe')
proc_path.mkdir(parents=True, exist_ok=True)
print(proc_path)

ap_file = next(session_path.joinpath('raw_ephys_data', pname).rglob('*.ap.cbin'))
lf_file = next(session_path.joinpath('raw_ephys_data', pname).rglob('*.lf.cbin'))

ap_file_destriped = proc_path.joinpath(ap_file.stem + ".bin")
lf_file_destriped = proc_path.joinpath(lf_file.stem + ".bin")

# sr = spikeglx.Reader(ap_file)
if not ap_file_destriped.exists():
    decompress_destripe_cbin(ap_file, output_file=ap_file_destriped, compute_rms=True, reject_channels=True)
    shutil.copy(ap_file.with_suffix(".meta"), ap_file_destriped.with_suffix(".meta"))

if not lf_file_destriped.exists():
    decompress_destripe_cbin(lf_file, output_file=lf_file_destriped, compute_rms=True, reject_channels=True,
                             butter_kwargs={'N': 3, 'Wn': 2 / 2500 * 2, 'btype': 'highpass'}, k_filter=False)
    shutil.copy(lf_file.with_suffix(".meta"), lf_file_destriped.with_suffix(".meta"))


sr_ = spikeglx.Reader(lf_file)
sr = spikeglx.Reader(lf_file_destriped)
t0 = 20

dat = sr[int(t0 * sr.fs): int((t0 + 0.5) * sr.fs), :]
dat_ = sr_[int(t0 * sr.fs): int((t0 + 0.5) * sr.fs), :]


from neurodsp.voltage import destripe

viewephys(dat.T, sr.fs, title='destripe')


viewephys(dat_.T, sr.fs, title='raw')
viewephys(destripe(dat_.T[:-1, :], sr.fs), sr.fs, title='destripe_')

