from pathlib import Path
import numpy as np
from neurodsp import voltage
from one.api import ONE
from brainbox.io.spikeglx import Streamer
from ephys_atlas.data import bwm_pids
import yaml
import scipy.signal

ROOT_PATH = Path("/mnt/s0/ephys-atlas")
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data

one = ONE(base_url="https://alyx.internationalbrainlab.org")
pids, _ = bwm_pids(one)


def stream_insertion_chunks(pid, typ='ap'):
    if typ == 'ap':
        sample_duration, sample_spacings, skip_start_end = (10 * 30_000, 1_000 * 30_000, 500 * 30_000)
        suffix = "destripe"
    elif typ == 'lf':
        sample_duration, sample_spacings, skip_start_end = (30 * 2_000, 1_000 * 2_000, 500 * 2_000)
        suffix = 'lfp'
    sr = Streamer(pid=pid, one=one, remove_cached=True, typ=typ)
    chunk_size = sr.chunks['chunk_bounds'][1]
    nsamples = np.ceil((sr.shape[0] - sample_duration - skip_start_end * 2) / sample_spacings)
    t0_samples = np.round((np.arange(nsamples) * sample_spacings + skip_start_end) / chunk_size) * chunk_size

    for s0 in t0_samples:
        path_pid = ROOT_PATH.joinpath('_'.join(([pid] + list(session_path.parts[-3:]) + [pname])))
        # ap: chunk1500_destripe.yml, lf: chunk1200_lfp.npy
        file_destripe = path_pid.joinpath(f'{prefix}chunk{str(int(s0 / chunk_size)).zfill(4)}_{suffix}.npy')
        file_yaml = path_pid.joinpath(f'{prefix}chunk{str(int(s0 / chunk_size)).zfill(4)}_{suffix}.yml')
        if file_destripe.exists():
            continue
        print(f"{i}, {pid}, {typ}, {file_destripe.name}")
        tsel = slice(int(s0), int(s0) + int(sample_duration))
        raw = sr[tsel, :-sr.nsync].T
        if typ == 'ap':
            destripe = voltage.destripe(raw, fs=sr.fs, neuropixel_version=1)
            fs_out = sr.fs
        elif typ == 'lf':
            destripe = voltage.destripe_lfp(raw, fs=sr.fs, neuropixel_version=1)
            destripe = scipy.signal.decimate(destripe, LFP_RESAMPLE_FACTOR, axis=1, ftype='fir')
            fs_out = sr.fs / LFP_RESAMPLE_FACTOR
        # example: PosixPath('/mnt/s0/ephys-atlas/06d42449-d6ac-4c35-8f85-24ecfbc08bc1_ibl_witten_29_2021-06-10_003_probe00')
        path_pid.mkdir(exist_ok=True)
        np.save(file_destripe, destripe.astype(np.float16))
        with open(file_yaml, 'w+') as fp:
            yaml.dump(dict(fs=fs_out, eid=eid, pid=pid, pname=pname, nc=raw.shape[0], dtype="float16"), fp)


IMIN = 582
for i, pid in enumerate(pids):
    if i < IMIN:
        continue
    eid, pname = one.pid2eid(pid)
    session_path = one.eid2path(eid)
    stream_insertion_chunks(pid, typ='ap')
    stream_insertion_chunks(pid, typ='lf')


#eid = 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', session_path="PosixPath('/mnt/s0/Data/wittenlab/Subjects/ibl_witten_13/2019-12-03/001'): multiple meta files found

## this snippet was to rewrite chunks as float16
# from pathlib import Path
# ROOT_PATH = Path("/mnt/s0/ephys-atlas")
# for file_destripe in ROOT_PATH.rglob("*_destripe.npy"):
#     file_yaml = file_destripe.with_suffix('.yml')
#     with open(file_yaml) as fp:
#         finfo = yaml.safe_load(fp)
#     if finfo['dtype'] == 'float32':
#         print(file_yaml)
#     else:
#         continue
#     finfo['dtype'] = 'float16'
#     np.save(file_destripe, np.load(file_destripe).astype(np.float16))
#     with open(file_yaml, 'w+') as fp:
#         yaml.dump(finfo, fp)



# '11a5a93e-58a9-4ed0-995e-52279ec16b98' multiple objects found for Streamer