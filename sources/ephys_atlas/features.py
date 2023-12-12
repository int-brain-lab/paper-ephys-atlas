import numpy as np
import pandas as pd
import scipy.signal

from neurodsp.utils import rms, fcn_cosine
import neuropixel
from neurodsp.waveforms import compute_spike_features

BANDS = {'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12], 'beta': [15, 30], 'gamma': [30, 90], 'lfp': [0, 90]}


def _get_power_in_band(fscale, period, band):
    band = np.array(band)
    # weight the frequencies
    fweights = fcn_cosine([-np.diff(band), 0])(-abs(fscale - np.mean(band)))
    p = 10 * np.log10(np.sum(period * fweights / np.sum(fweights), axis=-1))  # dB relative to v/sqrt(Hz)
    return p


def lf(data, fs, bands=None):
    """
    Computes the LF features from a numpy array
    :param data: numpy array with the data (channels, samples)
    :param fs: sampling interval (Hz)
    :param bands: dictionary with the bands to compute (default: BANDS constant)
    :return: pandas dataframe with the columns ['channel', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
       'psd_gamma', 'psd_lfp']
    """
    bands = BANDS if bands is None else bands
    nc = data.shape[0]  # number of channels
    fscale, period = scipy.signal.periodogram(data, fs)
    df_chunk = pd.DataFrame()
    df_chunk['channel'] = np.arange(nc)
    df_chunk['rms_lf'] = rms(data, axis=-1)
    for b in BANDS:
        df_chunk[f"psd_{b}"] = _get_power_in_band(fscale, period, bands[b])
    return df_chunk


def ap(data):
    """
    Computes the LF features from a numpy array
    :param data: numpy array with the AP band data (channels, samples)
    :return: pandas dataframe with the columns ['channel', 'rms_ap']
    """
    df_chunk = pd.DataFrame()
    nc = data.shape[0]  # number of channels
    df_chunk['channel'] = np.arange(nc)
    df_chunk['rms_ap'] = rms(data, axis=-1)
    return df_chunk


def spikes(data, fs, h=None):
    import spike_psvae.subtract  # needs the numpy_subtract branch
    h = neuropixel.trace_header(version=1) if h is None else h
    zdata = data / rms(data, axis=-1)[:, np.newaxis]
    TROUGH_OFFSET = 42
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
    df_spikes, waveforms = spike_psvae.subtract.subtract_and_localize_numpy(zdata.T.astype(np.float32), geom, **kwargs)
    df_waveforms = compute_spike_features(np.array(waveforms))
    df_spikes = df_spikes.merge(df_waveforms, left_index=True, right_index=True)
    df_spikes['channel'] = df_spikes['trace'].astype(np.int16)

    fcn_mean_time = lambda x: np.mean((x - TROUGH_OFFSET)) / fs

    # aggregation by channel of the spikes / waveforms features
    df_channels = df_spikes.groupby('channel').agg(
        alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha", aggfunc=lambda x: np.std(x, ddof=0)),
        spike_count=pd.NamedAgg(column="alpha", aggfunc="count"),
        peak_time_secs=pd.NamedAgg(column="peak_time_idx", aggfunc=fcn_mean_time),
        peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
        trough_time_secs=pd.NamedAgg(column="trough_time_idx", aggfunc=fcn_mean_time),
        trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
        tip_time_secs=pd.NamedAgg(column="tip_time_idx", aggfunc=fcn_mean_time),
        tip_val=pd.NamedAgg(column="tip_val", aggfunc="mean"),
        recovery_time_secs=pd.NamedAgg(column="recovery_time_idx", aggfunc=fcn_mean_time),
        depolarisation_slope=pd.NamedAgg(column="depolarisation_slope", aggfunc="mean"),
        repolarisation_slope=pd.NamedAgg(column="repolarisation_slope", aggfunc="mean"),
        recovery_slope=pd.NamedAgg(column="recovery_slope", aggfunc="mean"),
        polarity=pd.NamedAgg(column='invert_sign_peak', aggfunc=lambda x: -x.mean()),
    )

    return df_channels, df_spikes
