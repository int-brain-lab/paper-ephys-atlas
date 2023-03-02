from pathlib import Path
import traceback
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


_logger = setup_logger('ephys_atlas', level='INFO')

AP_RAW_TIMES = [0.5, 0.55]
LF_RAW_TIMES = [10, 10.5]
LFP_RESAMPLE_FACTOR = 10  # 200 Hz data
VERSION = '1.1.0'


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

    :param pid:
    :param one:
    :param typ:
    :param prefix:
    :param destination:
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
        butter_kwargs = {'N': 3, 'Wn': 2 / 2500 * 2, 'btype': 'highpass'}
        raw_sample_times = LF_RAW_TIMES
    sr = Streamer(pid=pid, one=one, remove_cached=remove_cached, typ=typ)
    chunk_size = sr.chunks['chunk_bounds'][1]
    nsamples = np.ceil((sr.shape[0] - sample_duration - skip_start_end * 2) / sample_spacings)
    t0_samples = np.round((np.arange(nsamples) * sample_spacings + skip_start_end) / chunk_size) * chunk_size
    sos = scipy.signal.butter(**butter_kwargs, output='sos')

    for s0 in t0_samples:
        t0 = int(s0 / chunk_size)
        file_destripe = destination.joinpath(f"T{t0:05d}", f"{typ}.npy")
        file_yaml = file_destripe.with_suffix('.yml')
        if file_destripe.exists() and clobber is False:
            continue
        tsel = slice(int(s0), int(s0) + int(sample_duration))
        raw = sr[tsel, :-sr.nsync].T
        butt = scipy.signal.sosfiltfilt(sos, raw)[:, int(sr.fs * raw_sample_times[0]):int(sr.fs * raw_sample_times[1])]
        if typ == 'ap':
            destripe = voltage.destripe(raw, fs=sr.fs, neuropixel_version=1, channel_labels=True)
            # saves a 0.05 secs snippet of the butterworth filtered data at 0.5sec offset for QC purposes
            fs_out = sr.fs
        elif typ == 'lf':
            destripe = voltage.destripe_lfp(raw, fs=sr.fs, neuropixel_version=1, channel_labels=True)
            destripe = scipy.signal.decimate(destripe, LFP_RESAMPLE_FACTOR, axis=1, ftype='fir')
            butt = scipy.signal.decimate(destripe, LFP_RESAMPLE_FACTOR, axis=1, ftype='fir')
            fs_out = sr.fs / LFP_RESAMPLE_FACTOR
        file_destripe.parent.mkdir(exist_ok=True, parents=True)
        np.save(file_destripe, destripe.astype(np.float16))
        with open(file_yaml, 'w+') as fp:
            yaml.dump(dict(fs=fs_out, eid=eid, pid=pid, pname=pname, nc=raw.shape[0], dtype="float16"), fp)
        if typ == 'ap':
            np.save(file_destripe.parent.joinpath('raw.npy'), butt.astype(np.float16))


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
        trough_offset=42,
        spike_length_samples=121,
        loc_workers=1
    )
    # channel_index = make_channel_index(geom, kwargs['extract_radius'], distance_order=False)

    all_files = list(destination.rglob('ap.npy'))
    for i, ap_file in enumerate(all_files):
        chunk_dir = ap_file.parent
        file_error = chunk_dir.joinpath('error_localisation.txt')
        file_waveforms = chunk_dir.joinpath('waveforms.npy')
        file_spikes = chunk_dir.joinpath('spikes.pqt')
        if file_error.exists():
            _logger.info(f"{i}/{len(all_files)}: {ap_file}, SKIP PREVIOUS ERROR")
            continue
        if file_waveforms.exists() and file_spikes.exists() and clobber is False:
            _logger.info(f"{i}/{len(all_files)}: {ap_file}, SKIP")
            continue
        _logger.info(f"{i}/{len(all_files)}: {ap_file}, COMPUTE")
        data = np.load(ap_file).astype(np.float32)
        # here the normalisation is based off a single chunk, but should this be constant for the whole recording ?
        data = data / rms(data, axis=-1)[:, np.newaxis]
        wg = WindowGenerator(data.shape[-1], 30000, overlap=0)
        localisation = []
        try:
            for first, last in wg.firstlast:
                loc, wfs = subtract_and_localize_numpy(data[:, first:last].T, geom, **kwargs)
                cleaned_wfs = wfs if first == 0 else np.concatenate([cleaned_wfs, wfs], axis=0)
                loc['sample'] += first
                localisation.append(loc)
        except (TypeError, ValueError) as e:
            errstr = traceback.format_exc()
            _logger.error(f"type error: {ap_file}, {errstr}")
            with open(file_error, 'w+') as fp:
                fp.write(errstr)
            continue
        localisation = pd.concat(localisation)
        np.save(file_waveforms, cleaned_wfs)
        localisation.to_parquet(file_spikes)


def get_raw_waveform(data, h, df_spikes, iw, trough_offset=42, spike_length_samples=121, extract_radius=200):
    xy = h['x'] + 1j * h['y']
    s0 = int(df_spikes['sample'].iloc[iw] - trough_offset)
    sind = slice(s0, s0 + int(spike_length_samples))
    cind = np.abs(xy[int(df_spikes['trace'].iloc[iw])] - xy) <= extract_radius
    hwav = {k: v[cind] for k, v in h.items()}
    return data[cind, sind].T, hwav
