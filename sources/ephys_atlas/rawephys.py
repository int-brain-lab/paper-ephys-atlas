import yaml

import numpy as np
import scipy.signal
import pandas as pd

import neuropixel
from neurodsp import voltage
from neurodsp.utils import rms
from brainbox.io.spikeglx import Streamer

from iblutil.util import setup_logger
from neurodsp.utils import WindowGenerator
from neurodsp.waveforms import compute_spike_features
from neurodsp.voltage import current_source_density
from neurodsp.cadzow import cadzow_np1
from neuropixel import trace_header
import ephys_atlas.features

_logger = setup_logger('ephys_atlas', level='INFO')

AP_RAW_TIMES = [5., 6.]
LFP_RESAMPLE_FACTOR = 5  # 200 Hz data
VERSION = '1.3.0'
TROUGH_OFFSET = 42


def destripe(pid, one=None, typ='ap', prefix="", destination=None, remove_cached=True, clobber=False):
    """
    Stream chunks of data from a given probe insertion

    Output folder architecture (the UUID is the probe insertion UUID):
        f4bd76a6-66c9-41f3-9311-6962315f8fc8
        ├── T00500
        │   ├── ap.npy
        │   ├── ap.yml
        │   ├── lf.npy
        │   ├── lf.yml
        │   ├── spikes.pqt
        │   └── waveforms.npy

    :param pid: probe insertion UUID
    :param one: one.api.ONE instance
    :param typ: frequency band ("ap" or "lf")
    :param prefix:
    :param destination: Path to save data
    :return:
    """
    assert one
    assert destination
    eid, pname = one.pid2eid(pid)

    if typ == 'ap':
        sample_duration, sample_spacings, skip_start_end = (10 * 30_000, 1_000 * 30_000, 500 * 30_000)
        butter_kwargs = {'N': 3, 'Wn': 300 / 30000 * 2, 'btype': 'highpass'}
        raw_sample_times = AP_RAW_TIMES
    elif typ == 'lf':
        sample_duration, sample_spacings, skip_start_end = (20 * 2_500, 1_000 * 2_500, 500 * 2_500)
        butter_kwargs = {'N': 3, 'Wn': [2 / 2500 * 2, 200 / 2500 * 2], 'btype': 'bandpass'}
        raw_sample_times = [0, sample_duration]
    sr = Streamer(pid=pid, one=one, remove_cached=remove_cached, typ=typ)
    chunk_size = sr.chunks['chunk_bounds'][1]
    nsamples = np.ceil((sr.shape[0] - sample_duration - skip_start_end * 2) / sample_spacings)
    assert nsamples > 0, f"Recording length is too small {sr.shape[0] / sr.fs: 0.2f} secs"
    t0_samples = np.round((np.arange(nsamples) * sample_spacings + skip_start_end) / chunk_size) * chunk_size
    sos = scipy.signal.butter(**butter_kwargs, output='sos')

    for s0 in t0_samples:
        t0 = int(s0 / chunk_size)
        file_destripe = destination.joinpath(f"T{t0:05d}", f"{typ}_destripe.npy")
        file_yaml = file_destripe.with_suffix('.yml')
        if file_destripe.exists() and clobber is False:
            continue
        tsel = slice(int(s0), int(s0) + int(sample_duration))
        raw = sr[tsel, :-sr.nsync].T
        if typ == 'ap':
            destripe = voltage.destripe(raw, fs=sr.fs, neuropixel_version=1, channel_labels=True)
            # saves a 0.05 secs snippet of the butterworth filtered data at 0.5sec offset for QC purposes
            fs_out = sr.fs
        elif typ == 'lf':
            destripe = voltage.destripe_lfp(raw, fs=sr.fs, neuropixel_version=1, channel_labels=True)
            destripe = scipy.signal.decimate(destripe, LFP_RESAMPLE_FACTOR, axis=1, ftype='fir')
            raw = scipy.signal.decimate(raw, LFP_RESAMPLE_FACTOR, axis=1, ftype='fir')
            fs_out = sr.fs / LFP_RESAMPLE_FACTOR
        file_destripe.parent.mkdir(exist_ok=True, parents=True)
        np.save(file_destripe, destripe.astype(np.float16))
        np.save(file_destripe.parent.joinpath(f'{typ}_raw.npy'),
                raw.astype(np.float16)[:, int(sr.fs * raw_sample_times[0]):int(sr.fs * raw_sample_times[1])]
                )
        with open(file_yaml, 'w+') as fp:
            yaml.dump(dict(fs=fs_out, eid=eid, pid=pid, pname=pname, nc=raw.shape[0], dtype="float16"), fp)
        


def localisation(destination=None, clobber=False):
    """
    :return:
    """
    from spike_psvae.subtract import subtract_and_localize_numpy
    h = neuropixel.trace_header(version=1)
    geom = np.c_[h['x'], h['y']]

    kwargs = dict(
        extract_radius=200.,
        loc_radius=100.,
        dedup_spatial_radius=70.,
        thresholds=[12, 10, 8, 6, 5],
        radial_parents=None,
        tpca=None,
        device=None,
        probe="np1",
        trough_offset=TROUGH_OFFSET,
        spike_length_samples=121,
        loc_workers=1
    )
    # channel_index = make_channel_index(geom, kwargs['extract_radius'], distance_order=False)

    all_files = list(destination.rglob('ap_destripe.npy'))
    for i, ap_file in enumerate(all_files):
        chunk_dir = ap_file.parent
        file_waveforms = chunk_dir.joinpath('waveforms.npy')
        file_spikes = chunk_dir.joinpath('spikes.pqt')
        if file_waveforms.exists() and file_spikes.exists() and clobber is False:
            _logger.info(f"{i}/{len(all_files)}: {ap_file}, SKIP")
            continue
        _logger.info(f"{i}/{len(all_files)}: {ap_file}, COMPUTE")
        data = np.load(ap_file).astype(np.float32)
        # here the normalisation is based off a single chunk, but should this be constant for the whole recording ?
        data = data / rms(data, axis=-1)[:, np.newaxis]
        wg = WindowGenerator(data.shape[-1], 30000, overlap=0)
        localisation = []
        for first, last in wg.firstlast:
            loc, wfs = subtract_and_localize_numpy(data[:, first:last].T, geom, **kwargs)
            cleaned_wfs = wfs if first == 0 else np.concatenate([cleaned_wfs, wfs], axis=0)
            loc['sample'] += first
            localisation.append(loc)
        localisation = pd.concat(localisation).reset_index()
        np.save(file_waveforms, cleaned_wfs)
        localisation.to_parquet(file_spikes)


def get_raw_waveform(data, h, df_spikes, iw, trough_offset=42, spike_length_samples=121, extract_radius=200):
    xy = h['x'] + 1j * h['y']
    s0 = int(df_spikes['sample'].iloc[iw] - trough_offset)
    sind = slice(s0, s0 + int(spike_length_samples))
    cind = np.abs(xy[int(df_spikes['trace'].iloc[iw])] - xy) <= extract_radius
    hwav = {k: v[cind] for k, v in h.items()}
    return data[cind, sind].T, hwav


def compute_ap_features(pid, root_path=None):
    """
    Reads in the destriped APs and computes the AP features
    :param pid, root_path:
    :return: Dataframe with the AP features:
        -   rms_ap (V): RMS of the AP band
    """
    assert root_path
    pfolder = root_path.joinpath(pid)
    files_destripe = list(pfolder.rglob('ap_destripe.npy'))
    nfiles = len(files_destripe)
    assert nfiles > 0, 'error: no AP destripe chunk found !'
    df_chunks = []
    rl = 0
    for i in np.arange(nfiles):
        file_destripe = files_destripe[i]
        with open(file_destripe.with_suffix('.yml')) as fp:
            ap_info = yaml.safe_load(fp)
        data = np.load(file_destripe).astype(np.float32)
        df_chunk = ephys_atlas.features.ap(data)
        df_chunks.append(df_chunk)
        rl += data.shape[1] / ap_info['fs']
    if len(df_chunks) == 0:
        return None, None
    df_chunks = pd.concat(df_chunks)
    ap_features = df_chunks.groupby('channel').agg(
        rms_ap=pd.NamedAgg(column="rms_ap", aggfunc="mean"),
    )
    return ap_features, ap_info['fs']


def compute_lf_features(pid, root_path=None, bands=None, current_source=False):
    """
    Reads in the destriped LF and computes the LF features
    :param pid, root_path:
    :param csd: False: if set to True, computes current source density from the RMS trace
    :return: Dataframe with the LF features:
        -   rms_lf (V): RMS of the LF band
        -   psd_delta (dB rel V ** 2 / Hz): Power in the delta band (also theta, alpha, beta, gamma)
    """
    pfolder = root_path.joinpath(pid)
    files_lfp = list(pfolder.rglob('lf_destripe.npy'))
    nfiles = len(files_lfp)
    assert nfiles > 0, f'error: no LFP destripe chunk found for pid {pid}!'
    df_chunks = []
    for i in np.arange(nfiles):
        file_lfp = files_lfp[i]
        with open(file_lfp.with_suffix('.yml')) as fp:
            lf_info = yaml.safe_load(fp)
        # loads the LFP and compute spectra for each channel
        data = np.load(file_lfp).astype(np.float32)
        if current_source:
            h = trace_header(version=1)
            cadzow = cadzow_np1(data, rank=2, fs=250, niter=1)
            data = current_source_density(cadzow, h=h)
        df_chunk = ephys_atlas.features.lf(data, lf_info['fs'], bands=ephys_atlas.features.BANDS)
        df_chunks.append(df_chunk)

    df_chunks = pd.concat(df_chunks)
    lf_features = df_chunks.groupby('channel').agg(
        rms_lf=pd.NamedAgg(column="rms_lf", aggfunc="median"),
        **{f"psd_{b}": pd.NamedAgg(column=f"psd_{b}", aggfunc="median") for b in ephys_atlas.features.BANDS}
    )
    return lf_features


def compute_spikes_features(pid, root_path=None):
    """
    Reads in the spikes parquet file and computes spikes features
    :param pid, root_path:
    :return: Dataframe with spikes features:
            'sample'
             'trace'
             'x', 'y', 'z', 'alpha'
             't0'
             'peak_trace_idx'
             'peak_time_idx'
             'peak_val'
             'trough_time_idx'
             'trough_val'
       'tip_time_idx', 'tip_val']
    """
    assert root_path
    pfolder = root_path.joinpath(pid)
    files_spikes = list(pfolder.rglob('spikes.pqt'))
    nfiles = len(files_spikes)
    assert len(files_spikes) > 0, f"No localisation chunk found pid {pid}!"
    df_spikes = []
    for i in np.arange(nfiles):
        file_spikes = files_spikes[i]
        file_waveforms = file_spikes.with_name('waveforms.npy')
        waveforms = np.load(file_waveforms)
        df_wav = compute_spike_features(waveforms)
        df_tmp = pd.read_parquet(file_spikes)
        df_tmp['t0'] = int(file_spikes.parts[-2][1:])
        df_spikes.append(df_tmp.merge(df_wav, left_index=True, right_index=True))
    df_spikes = pd.concat(df_spikes)
    return df_spikes
