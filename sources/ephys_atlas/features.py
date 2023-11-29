import numpy as np
import pandas as pd
import scipy.signal

from neurodsp.utils import rms, fcn_cosine

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
