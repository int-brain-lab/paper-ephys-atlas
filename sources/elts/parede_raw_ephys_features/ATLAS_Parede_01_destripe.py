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

pids.sort()


c = 0
IMIN = 486
for i, pid in enumerate(pids):
    if i < IMIN:
        continue
    print(i, pid)
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf', clobber=False)
    for flag in destination.glob('.01_destripe_1*'):
        flag.unlink()
    if destination.exists():
        destination.joinpath(f'.01_destripe_{VERSION}').touch()

## %%
"""
This part recovers a snippet of butterworth filtered data for the sole purpose of visual destriping QC
"""
pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
    '5f7766ce-8e2e-410c-9195-6bf089fea4fd',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]
import scipy
from brainbox.io.spikeglx import Streamer
import numpy as np
butter_kwargs = {'N': 3, 'Wn': 300 / 30000 * 2, 'btype': 'highpass'}
sos = scipy.signal.butter(**butter_kwargs, output='sos')

c = 0
IMIN = 0
for i, pid in enumerate(pids):
    if i < IMIN:
        continue
    destination = ROOT_PATH.joinpath(pid)
    for ap_file in destination.rglob('ap.npy'):
        raw_file = ap_file.parent.joinpath('raw.npy')
        if raw_file.exists():
            print(i, pid, ap_file, 'CONTINUE')
            continue
        print(i, pid, ap_file, 'PROCESS')

        t0 = int(ap_file.parts[-2][1:])
        sr = Streamer(pid=pid, one=one, remove_cached=True, typ='ap')
        chunk_size = sr.chunks['chunk_bounds'][1]
        s0 = t0 * chunk_size
        tsel = slice(int(s0), int(s0) + int(30000))
        raw = sr[tsel, :-sr.nsync].T
        # saves a 0.05 secs snippet of the butterworth filtered data at 0.5sec offset for QC purposes
        butt = scipy.signal.sosfiltfilt(sos, raw)[:, int(sr.fs * 0.5):int(sr.fs * 0.55)]
        np.save(ap_file.parent.joinpath('raw.npy'), butt.astype(np.float16))
