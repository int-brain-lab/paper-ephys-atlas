import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurodsp.utils import rms
import neuropixel
from ibllib.plots import wiggle
from viewephys.gui import viewephys

ROOT_PATH = Path("/mnt/s0/ephys-atlas")

h = neuropixel.trace_header(version=1)
geom = np.c_[h['x'], h['y']]
xy = h['x'] + 1j * h['y']

kwargs = dict(
    extract_radius=200.,
    loc_radius = 100.,
    dedup_spatial_radius = 70.,
    thresholds=[12, 10, 8, 6, 5],
    radial_parents=None,
    tpca=None,
    device=None,
    probe="np1",
    trough_offset=42,
    spike_length_samples=121,
    loc_workers=1
)


## Load a recording
if sys.platform == 'linux':
    file_destripe = Path("/datadisk/scratch/ephys-atlas/06d42449-d6ac-4c35-8f85-24ecfbc08bc1_ibl_witten_29_2021-06-10_003_probe00/t0500_destripe.npy")
    file_destripe = Path("/datadisk/scratch/ephys-atlas/f336f6a4-f693-4b88-b12c-c5cf0785b061_KS096_2022-06-21_001_probe00/chunk0500_destripe.npy")
else:
    file_destripe = Path("/Users/olivier/Documents/datadisk/atlas/f336f6a4-f693-4b88-b12c-c5cf0785b061_KS096_2022-06-21_001_probe00/chunk0500_destripe.npy")

for file_destripe in ROOT_PATH.rglob('*_destripe.npy'):
    print(file_destripe)
    continue

data = np.load(file_destripe).astype(np.float32)
data = data / rms(data, axis=-1)[:, np.newaxis]

df_spikes = pd.read_parquet(str(file_destripe).replace('_destripe.npy', '_spikes.pqt'))
waveforms = np.load(str(file_destripe).replace('_destripe.npy', '_waveforms.npy'))
df_spikes['r'] = np.abs(xy[df_spikes['trace'].values.astype(np.int32)] - df_spikes['x'] - 1j * df_spikes['y'])

with open(file_destripe.with_suffix('.yml')) as fp:
    rinfo = yaml.safe_load(fp)
# df_spikes.describe Index(['sample', 'trace', 'x', 'y', 'z', 'alpha'], dtype='object')


## Load a given spike
iw = 520

s0 = int(df_spikes['sample'].iloc[iw] - kwargs['trough_offset'])
sind = slice(s0, s0 + int(kwargs['spike_length_samples']))
cind = np.abs(xy[int(df_spikes['trace'].iloc[iw])] - xy) <= kwargs['extract_radius']
hwav = {k: v[cind] for k, v in h.items()}

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
wiggle(- np.squeeze(waveforms[iw, :, :]), fs=rinfo['fs'], gain=0.25, ax=axs[0])
wiggle(- data[cind, sind].T, fs=rinfo['fs'], gain=.25, ax=axs[1])
wiggle(- np.squeeze(waveforms[iw, :, :]) + data[cind, sind].T, fs=rinfo['fs'], gain=.25, ax=axs[2])


##

wc = viewephys(waveforms[iw, :, :].T, fs=rinfo['fs'], title='wav_clean', channels=hwav)
wr = viewephys(data[cind, sind], fs=rinfo['fs'], title='wav_raw')

## Display detections
fs = rinfo['fs']
from viewephys.gui import viewephys
eqcs = {}
eqcs[t] = viewephys(data[:, :20000], fs=fs, title=(t := 'destripe'), a_scalar=1)

for k, eqc in eqcs.items():
    eqc.ctrl.add_scatter(df_spikes['sample'].values / fs * 1e3, df_spikes['trace'].values, label='loc')
