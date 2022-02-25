from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal

from brainbox.io.spikeglx import stream

from iblutil.numerical import ismember2d
from ibllib.dsp import voltage
from ibllib.atlas.regions import BrainRegions
import ibllib.dsp as dsp
from ibllib.ephys.neuropixel import trace_header
from ibllib.ephys.spikes import detection

from one.api import ONE

h = trace_header(version=1)
br = BrainRegions()
one = ONE()
V_T0 = [60*10, 60*30, 60*50]  # sample at 10, 30, 50 min in
STAGING_PATH = Path('/datadisk/FlatIron/tables/bwm')

df_channels = pd.read_parquet(STAGING_PATH.joinpath('channels.pqt'))
df_clusters = pd.read_parquet(STAGING_PATH.joinpath('clusters.pqt'))
df_probes = pd.read_parquet(STAGING_PATH.joinpath('probes.pqt'))

pid = df_probes.index[0]
# Run on the full file
excludes = [
     '5b489aee-75e8-42af-aca6-3026d9a54c2d',  # lust ubdex iyt if range   i0 = cmeta['chunk_bounds'][first_chunk]
]

BANDS = {'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12], 'beta': [15, 30], 'gamma': [30, 90]}

new_fields = [f"psd_{k}" for k in BANDS] + ['rms_ap', 'rms_lf', 'bad_channel', 'spike_rate']


def get_power_in_band(fscale, period, band):
    band = np.array(band)
    # weight the frequencies
    fweights = dsp.fcn_cosine([-np.diff(band), 0])(-abs(fscale - np.mean(band)))
    p = 10 * np.log10(np.sum(period * fweights / np.sum(fweights), axis=-1))  # # dB relative to v/sqrt(Hz)
    return p

#  (approx. 0-4 Hz), theta (4-10), alpha (8-12 Hz), beta (15-30 Hz) and gamma (30-90 Hz)
JMIN = 239
for j, pid in enumerate(df_probes.index):
    if pid not in df_channels.index:
        continue
    if pid in excludes:
        continue
    if j < JMIN:
        continue
    print(j, pid)
    chins = df_channels.loc[pid]
    nch = h['x'].size
    features_tmp = {k: np.zeros((nch, len(V_T0))) for k in new_fields}
    _, idf = ismember2d(chins.loc[:, ['lateral_um', 'axial_um']].to_numpy(), np.c_[h['x'], h['y']])


    for i, T0 in enumerate(V_T0):
        sr, t0 = stream(pid, T0, nsecs=1, one=one, typ='lf')
        sr_ap, _ = stream(pid, T0, nsecs=1, one=one)
        raw = sr[:, :-sr.nsync].T
        raw_ap = sr_ap[:, :-sr_ap.nsync].T
        [nc, ns] = raw.shape

        butter_kwargs = {'N': 3, 'Wn': 2 / sr.fs * 2, 'btype': 'highpass'}
        k_kwargs_car = {'ntr_pad': 60, 'ntr_tap': 0, 'lagc': None,
                        'butter_kwargs': {'N': 3, 'Wn': 0.001, 'btype': 'highpass'}}
        channel_labels, channel_features = voltage.detect_bad_channels(raw_ap, fs=sr_ap.fs)
        destripe = voltage.destripe(raw, sr.fs, butter_kwargs=butter_kwargs, channel_labels=channel_labels,
                               k_kwargs=k_kwargs_car, k_filter=False)
        destripe_ap = voltage.destripe(raw_ap, fs=sr_ap.fs, channel_labels=channel_labels,
                                       butter_kwargs={'N': 3, 'Wn': np.array([300, 5000]) / sr_ap.fs * 2, 'btype': 'bandpass'})
        fscale, period = scipy.signal.periodogram(destripe, sr.fs)
        p = {}
        for b in BANDS:
            features_tmp[f"psd_{b}"][:, i] = get_power_in_band(fscale, period, BANDS[b])
        features_tmp['bad_channel'][:, i] = channel_labels
        features_tmp['rms_ap'][:, i] = dsp.rms(destripe_ap)
        features_tmp['rms_lf'][:, i] = dsp.rms(destripe)
        spikes = detection(destripe_ap.T, sr_ap.fs, h, detect_threshold=-.00004, time_tol=.002, distance_threshold_um=70)
        nspikes = pd.DataFrame(spikes).groupby('trace')['time'].count()
        _, ifeat, id = np.intersect1d(np.arange(destripe_ap.shape[0]), nspikes.index.to_numpy(), return_indices=True)
        features_tmp['spike_rate'][ifeat, i] = nspikes.iloc[id]
        features_tmp['rms_lf'][:, i] = dsp.rms(destripe)
    for k in features_tmp:
        if k == 'bad_channel':
            mod, _ = scipy.stats.mode(features_tmp[k], axis=1)
            df_channels.loc[pid, k] = mod[idf]
        else:
            df_channels.loc[pid, k] = np.median(features_tmp[k], axis=-1)[idf]

df_channels.to_parquet(STAGING_PATH.joinpath('channels_voltage_features.pqt'))
#
# eqc_ap = viewephys(destripe_ap, fs=sr_ap.fs, title='ap', channels=df_channels.loc[pid], br=br)
# eqc_ap.ctrl.add_scatter(x=spikes.time * 1e3, y=spikes.trace, label='detection')
# eqc_lf = viewephys(destripe, fs=sr.fs, title='lfp', channels=df_channels.loc[pid], br=br)

# delta (approx. 0-4 Hz), theta (4-10), alpha (8-12 Hz), beta (15-30 Hz) and gamma (30-90 Hz)

def current_source_density(data, h):
    # data (nc, ns)
    # cds = current_source_density(data, h)
    def double_diff(data, ind):
        dx = np.diff(np.abs(h['x'][ind] + 1j * h['y'][ind]))
        tmp = np.diff(data[ind, :], axis=0) / dx[:, np.newaxis]
        tmp = np.diff(tmp, axis=0) / np.diff(dx)[:, np.newaxis]
        return np.r_[tmp[:1, :], tmp, tmp[-1:, :]]

    data[h['col'] >= 2, :] = double_diff(data, h['col'] >= 2)
    data[h['col'] < 2, :] = double_diff(data, h['col'] < 2)

    return data
