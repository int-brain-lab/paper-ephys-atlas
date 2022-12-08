'''
Stream LF data
Compute phase
'''

from one.api import ONE
from brainbox.io.spikeglx import Streamer
from neurodsp.voltage import destripe
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
from scipy import signal
from viewephys.gui import viewephys
import numpy as np
from numpy import angle

one = ONE()
ba = AllenAtlas()

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
pid = pids[0]

# == LOAD LFP

# Get the 1s of LFP data around time point of interest
t0 = 10  # timepoint in recording to stream
time_win = 1  # number of seconds to stream
band = 'lf'  # either 'ap' or 'lf'

sr = Streamer(pid=pid, one=one, remove_cached=False, typ=band)
s0 = t0 * sr.fs
tsel = slice(int(s0), int(s0) + int(time_win * sr.fs))
# remove sync channel from raw data
raw = sr[tsel, :-sr.nsync].T
# apply destriping algorithm to data
destriped_LFP = destripe(raw, fs=sr.fs)

# == Band-pass LF
fNQ = sr.fs/2
Wn = [31, 33]                          # Set the passband [5-7] Hz,
n = 100                             # ... and filter order,
b = signal.firwin(n, Wn, nyq=fNQ, pass_zero=False, window='hamming')
v_filt = signal.filtfilt(b, 1, destriped_LFP)    # ... and apply it to the data.

# compute phase
phi = angle(signal.hilbert(v_filt))

# == LOAD SPIKES
sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
# Do synching
from scipy.interpolate import interp1d
eid, probe = one.pid2eid(pid)
sync = one.load_dataset(eid, dataset='*.sync.npy', collection=f'raw_ephys_data/{probe}')
fcn_time_session2bin = interp1d(sync[:, 1], sync[:, 0], fill_value='extrapolate')
t0_bin = fcn_time_session2bin(t0)

slice_spikes = slice(np.searchsorted(spikes.times, t0_bin), np.searchsorted(spikes.times, t0_bin + time_win))
t = (spikes.times[slice_spikes] - t0_bin) * 1e3
c = clusters.channels[spikes.clusters[slice_spikes]]

# == Launch viewephys
eqc = dict()

eqc['filtered'] = viewephys(v_filt, fs=sr.fs)
eqc['filtered'].ctrl.add_scatter(t, c, rgb=(255, 0, 50), label='all')

eqc['angle'] = viewephys(phi, fs=sr.fs)
eqc['angle'].ctrl.add_scatter(t, c, rgb=(255, 0, 50), label='all')

eqc['destripe'] = viewephys(destriped_LFP, fs=sr.fs)
eqc['destripe'].ctrl.add_scatter(t, c, rgb=(255, 0, 50), label='all')

# %gui qt


# Plot colored line
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
def plt_phase_axis(x, y, phasey, fig, axs, axs_id):
    # Create a continuous norm to map from data points to colors
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(phasey.min(), phasey.max())
    lc = LineCollection(segments, cmap='hsv', norm=norm)
    # Set the values used for colormapping
    lc.set_array(phasey)
    lc.set_linewidth(2)
    line = axs[axs_id].add_collection(lc)
    fig.colorbar(line, ax=axs[axs_id])

    axs[axs_id].set_xlim(x.min(), x.max())
    axs[axs_id].set_ylim(y.min(), y.max())
    # plt.clim(-np.pi, np.pi)
    plt.show()

# Plot 1 channel LFP
ch_plt = 10  # 55
smpl_start = int(500/1000 * sr.fs)
smpl_end = int(700/1000 * sr.fs)
v_plt = v_filt[ch_plt, smpl_start:smpl_end+2]
phi_plt = phi[ch_plt, smpl_start:smpl_end+2]
t_plt = np.linspace(0, smpl_end-smpl_start, len(v_plt))
plt.plot(t_plt, v_plt)            # Plot the LFP data,

# WARNING actually hilbert works only 1D
phi_plt = angle(signal.hilbert(v_plt))

# fake datapoint to force axis lim
phi_plt[-1] = -np.pi
phi_plt[-2] = np.pi
v_plt[-1] = 0
v_plt[-2] = 0

x = t_plt
y = v_plt
phasey = phi_plt
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

plt_phase_axis(x, y, phasey, fig, axs, axs_id=0)







