## %%
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import scipy.signal

from one.alf.spec import is_uuid_string
from neurodsp.utils import rms, fcn_cosine
import neuropixel
from iblutil.util import setup_logger

_logger = setup_logger('atlas')


ROOT_PATH = Path("/mnt/s0/ephys-atlas")
STAGING_PATH = Path('/mnt/s0/aggregates/bwm')

# ROOT_PATH = Path("/Users/olivier/Documents/datadisk/atlas")
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

def get_power_in_band(fscale, period, band):
    band = np.array(band)
    # weight the frequencies
    fweights = fcn_cosine([-np.diff(band), 0])(-abs(fscale - np.mean(band)))
    p = 10 * np.log10(np.sum(period * fweights / np.sum(fweights), axis=-1))  # # dB relative to v/sqrt(Hz)
    return p


def get_ap_features(pfolder):
    """
    :param pfolder:
    :return:
        ap_features.keys()
        Index(['trace', 'rms_ap', 'alpha_mean', 'alpha_std', 'spike_rate',
               'cloud_x_std', 'cloud_y_std', 'cloud_z_std', 'pid']
    """
    files_destripe = list(pfolder.rglob('*_destripe.npy'))
    nfiles = len(files_destripe)
    df_chunks = []
    df_spikes = []
    rl = 0
    for i in np.arange(nfiles):
        file_destripe = files_destripe[i]
        file_spikes = Path(str(file_destripe).replace('_destripe.npy', '_spikes.pqt'))
        if not file_spikes.exists():
            _logger.warning(f'skip {file_destripe}')
            continue
        with open(file_destripe.with_suffix('.yml')) as fp:
            ap_info = yaml.safe_load(fp)
        _logger.info(f"{k} {file_destripe}")
        data = np.load(file_destripe).astype(np.float32)
        df_chunk = pd.DataFrame()
        df_chunk['raw_ind'] = np.arange(ap_info['nc'])
        df_chunk['rms_ap'] = rms(data, axis=-1)
        df_chunks.append(df_chunk)
        df_spikes.append(pd.read_parquet(file_spikes))
        rl += data.shape[1] / ap_info['fs']
    if len(df_chunks) == 0:
        return None, None
    df_chunks = pd.concat(df_chunks)
    df_spikes = pd.concat(df_spikes)

    spike_features = df_spikes.groupby('trace').agg(
        alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha", aggfunc="std"),
        spike_rate=pd.NamedAgg(column="alpha", aggfunc="count"),
        cloud_x_std=pd.NamedAgg(column="x", aggfunc="std"),
        cloud_y_std=pd.NamedAgg(column="y", aggfunc="std"),
        cloud_z_std=pd.NamedAgg(column="z", aggfunc="std"),
    )
    spike_features['spike_rate'] = spike_features['spike_rate'] / rl
    ap_features = df_chunks.groupby('raw_ind').agg(
        rms_ap=pd.NamedAgg(column="rms_ap", aggfunc="mean"),
    )
    df_out = pd.merge(ap_features, spike_features, left_index=True, right_on='trace', how='left')
    return df_out


def get_lf_features(pfolder):
    """
    :param pfolder:
    :return:
    lf_features.keys()
        Index(['rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
            'psd_gamma', 'pid'],
      dtype='object')
    """
    BANDS = {'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12], 'beta': [15, 30], 'gamma': [30, 90]}
    files_lfp = list(pfolder.rglob('*_lfp.npy'))
    nfiles = len(files_lfp)
    df_chunks = []
    for i in np.arange(nfiles):
        file_lfp = files_lfp[i]
        with open(file_lfp.with_suffix('.yml')) as fp:
            lf_info = yaml.safe_load(fp)
        _logger.info(f"{k} {file_lfp}")
        # loads the LFP and compute spectra for each channel
        data = np.load(file_lfp).astype(np.float32)
        fscale, period = scipy.signal.periodogram(data, lf_info['fs'])
        df_chunk = pd.DataFrame()
        df_chunk['raw_ind'] = np.arange(lf_info['nc'])
        df_chunk['rms_lf'] = rms(data, axis=-1)
        for b in BANDS:
            df_chunk[f"psd_{b}"] = get_power_in_band(fscale, period, BANDS[b])
        df_chunks.append(df_chunk)

    df_chunks = pd.concat(df_chunks)
    lf_features = df_chunks.groupby('raw_ind').agg(
        rms_lf=pd.NamedAgg(column="rms_lf", aggfunc="median"),
        **{f"psd_{b}": pd.NamedAgg(column=f"psd_{b}", aggfunc="median") for b in BANDS}
    )
    return lf_features


## %%
"""
df_out.keys()
Out[77]: 
Index(['trace', 'rms_ap', 'alpha_mean', 'alpha_std', 'spike_rate',
       'cloud_x_std', 'cloud_y_std', 'cloud_z_std', 'rms_lf', 'psd_delta',
       'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'pid'],
      dtype='object')
"""
pid_folders = [item for item in ROOT_PATH.iterdir() if (item.is_dir() and is_uuid_string(item.name[:36]))]
all_channels = []
KMIN = 0
for k, pfolder in enumerate(pid_folders):
    if k < KMIN:
        continue
    pid = pfolder.name[:36]
    ap_features = get_ap_features(pfolder)
    lf_features = get_lf_features(pfolder)

    if ap_features.index.name != 'trace':
        ap_features = ap_features.set_index('trace')
    df_out = pd.merge(lf_features, ap_features, left_index=True, right_index=True)
    df_out['pid'] = pid
    all_channels.append(df_out)

all_channels = pd.concat(all_channels)

all_channels['raw_ind'] = all_channels.index.values.astype(np.int32)
all_channels = all_channels.set_index(['pid', 'raw_ind'])

all_channels.to_parquet(STAGING_PATH.joinpath('raw_ephys_features.pqt'))

print(f'aws s3 sync "{STAGING_PATH}" s3://ibl-brain-wide-map-private/aggregates/bwm')
