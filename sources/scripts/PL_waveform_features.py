#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: PL based on OW 2022-07-26_ATLAS_Parede_03_compute_features.py

"""
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurodsp.utils import rms
import neuropixel
from ibllib.plots import wiggle
from ibllib.atlas import AllenAtlas
from viewephys.gui import viewephys

#from os.path import join
#from scipy.optimize import curve_fit
#from brainbox.io.one import SpikeSortingLoader
#from brainbox.metrics.single_units import spike_sorting_metrics

#from one.api import ONE

#ba = AllenAtlas()
#one = ONE()

#####region selection? specific trajectory selection? NeuronQC? Other QC...etc??? 

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
    file_destripe = Path("/Users/petrinalau/int-brain-lab/Data/f336f6a4-f693-4b88-b12c-c5cf0785b061_KS096_2022-06-21_001_probe00/chunk0500_destripe.npy")
else:
    file_destripe = Path("/Users/petrinalau/int-brain-lab/Data/f336f6a4-f693-4b88-b12c-c5cf0785b061_KS096_2022-06-21_001_probe00/chunk0500_destripe.npy")

data = np.load(file_destripe).astype(np.float32)
data = data / rms(data, axis=-1)[:, np.newaxis]

df_spikes = pd.read_parquet(file_destripe.with_suffix('.pqt'))
waveforms = np.load(file_destripe.parent.joinpath(file_destripe.stem + '_waveforms.npy'))
df_spikes['r'] = np.abs(xy[df_spikes['trace'].values.astype(np.int32)] - df_spikes['x'] - 1j * df_spikes['y'])

with open(file_destripe.with_suffix('.yml')) as fp:
    rinfo = yaml.safe_load(fp)
# df_spikes.describe Index(['sample', 'trace', 'x', 'y', 'z', 'alpha'], dtype='object')


## Load a given spike
iw = 650

s0 = int(df_spikes['sample'][iw] - kwargs['trough_offset'])
sind = slice(s0, s0 + int(kwargs['spike_length_samples']))

r = np.abs(xy[int(df_spikes['trace'][iw])] - xy)  # r is distance to spike

cind = r <= kwargs['extract_radius'] #index of channel of the waveform inside the raw data on selected channel that we can extract waveform from 
hwav = {k: v[cind] for k, v in h.items()}

np.argmin(r) # the channel that contains a local min
np.argmin(r[cind]) # 

imax = np.argmin(r[cind])

hwav['x'][imax]

w = np.squeeze(waveforms[iw, :, imax])

plt.plot(w)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
wiggle(- np.squeeze(waveforms[iw, :, :]), fs=rinfo['fs'], gain=0.25, ax=axs[0])
wiggle(- data[cind, sind].T, fs=rinfo['fs'], gain=.25, ax=axs[1])


## Feature extraction 

# Get peak-to-trough ratio
pt_ratio = np.max(w) / np.abs(np.min(w))

# Get part of spike from trough to first peak after the trough and repolarization
peak_after_trough = np.argmax(w[np.argmin(w):]) + np.argmin(w)
repolarization = w[np.argmin(w):np.argmax(w[np.argmin(w):]) + np.argmin(w)]

# Get spike width in ms
peak_to_trough = ((np.argmax(w) - np.argmin(w)) / 30000) * 1000
spike_width = ((peak_after_trough - np.argmin(w)) / 30000) * 1000



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
