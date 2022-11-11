import argparse
from pathlib import Path
import sys
import yaml
import shutil

import numpy as np
import scipy.signal
import pandas as pd

import neuropixel
from neurodsp import voltage
from neurodsp.utils import rms
from brainbox.io.spikeglx import Streamer
from one.api import ONE

from iblutil.util import get_logger
from neurodsp.utils import WindowGenerator


_logger = get_logger('ephys_atlas', level='INFO')

LFP_RESAMPLE_FACTOR = 10  # 200 Hz data
VERSION = '1.0.0'


def destripe(pid, one=None, typ='ap', prefix="", destination=None):
    """
    Stream chunks of data from a given probe insertion

    Output folder architecture (the UUID is the probe insertion UUID):
        f4bd76a6-66c9-41f3-9311-6962315f8fc8_ZFM-02369_2021-05-19_001_probe00
            ├── chunk0400_lfp.npy
            ├── chunk0400_lfp.yml
            ├── chunk0500_destripe.npy
            ├── chunk0500_destripe.yml

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
        suffix = "destripe"
    elif typ == 'lf':
        sample_duration, sample_spacings, skip_start_end = (30 * 2_000, 1_000 * 2_000, 500 * 2_000)
        suffix = 'lfp'
    sr = Streamer(pid=pid, one=one, remove_cached=True, typ=typ)
    chunk_size = sr.chunks['chunk_bounds'][1]
    nsamples = np.ceil((sr.shape[0] - sample_duration - skip_start_end * 2) / sample_spacings)
    t0_samples = np.round((np.arange(nsamples) * sample_spacings + skip_start_end) / chunk_size) * chunk_size

    for s0 in t0_samples:
        # ap: chunk1500_destripe.yml, lf: chunk1200_lfp.npy
        file_destripe = destination.joinpath(f'{prefix}chunk{str(int(s0 / chunk_size)).zfill(4)}_{suffix}.npy')
        file_yaml = destination.joinpath(f'{prefix}chunk{str(int(s0 / chunk_size)).zfill(4)}_{suffix}.yml')
        if file_destripe.exists():
            continue
        tsel = slice(int(s0), int(s0) + int(sample_duration))
        raw = sr[tsel, :-sr.nsync].T
        if typ == 'ap':
            destripe = voltage.destripe(raw, fs=sr.fs, neuropixel_version=1, channel_labels=True)
            fs_out = sr.fs
        elif typ == 'lf':
            destripe = voltage.destripe_lfp(raw, fs=sr.fs, neuropixel_version=1, channel_labels=True)
            destripe = scipy.signal.decimate(destripe, LFP_RESAMPLE_FACTOR, axis=1, ftype='fir')
            fs_out = sr.fs / LFP_RESAMPLE_FACTOR
        # destination example: PosixPath('/mnt/s1/ephys-atlas/06d42449-d6ac-4c35-8f85-24ecfbc08bc1_ibl_witten_29_2021-06-10_003_probe00')
        destination.mkdir(exist_ok=True)
        np.save(file_destripe, destripe.astype(np.float16))
        with open(file_yaml, 'w+') as fp:
            yaml.dump(dict(fs=fs_out, eid=eid, pid=pid, pname=pname, nc=raw.shape[0], dtype="float16"), fp)


def localisation(destination=None):
    """
    :return:
    """
    from spike_psvae.subtract import make_channel_index, subtract_and_localize_numpy
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

    all_files = list(destination.rglob('chunk*_destripe.npy'))
    for i, file_destripe in enumerate(all_files):
        file_waveforms = Path(str(file_destripe).replace('_destripe.npy', '_waveforms.npy'))
        file_spikes = Path(str(file_destripe).replace('_destripe.npy', '_spikes.pqt'))
        if file_waveforms.exists() and file_spikes.exists():
            continue
        _logger.info(f"{i}/{len(all_files)}: {file_destripe}")
        data = np.load(file_destripe).astype(np.float32)
        # here the normalisation is based off a single chunk, but should this be constant for the whole recording ?
        data = data / rms(data, axis=-1)[:, np.newaxis]
        wg = WindowGenerator(data.shape[-1], 30000, overlap=0)
        localisation = []
        try:
            for first, last in wg.firstlast:
                loc, wfs = subtract_and_localize_numpy(data[:, first:last].T, geom, **kwargs)
                cleaned_wfs = wfs if first == 0 else np.concatenate([cleaned_wfs, wfs], axis=0)
                localisation.append(loc)
        except TypeError as e:
            _logger.error(f"type error: {file_destripe}")
            continue
        localisation = pd.concat(localisation)
        np.save(file_waveforms, cleaned_wfs)
        localisation.to_parquet(file_spikes)
