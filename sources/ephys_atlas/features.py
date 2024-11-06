from pathlib import Path
import random
import shutil
import string

import numpy as np
import pandas as pd
import scipy.signal
import pydantic

import ibldsp.waveforms
import ibldsp.utils


class DartParameters(pydantic.BaseModel):
    localization_radius: pydantic.PositiveFloat = 150
    chunk_length_samples: pydantic.PositiveInt = 2 ** 15
    trough_offset: pydantic.PositiveInt = 42,


BANDS = {'delta': [0, 4], 'theta': [4, 10], 'alpha': [8, 12], 'beta': [15, 30], 'gamma': [30, 90], 'lfp': [0, 90]}


def _get_power_in_band(fscale, period, band):
    band = np.array(band)
    # weight the frequencies
    fweights = ibldsp.utils.fcn_cosine([-np.diff(band), 0])(-abs(fscale - np.mean(band)))
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
    df_chunk['rms_lf'] = ibldsp.utils.rms(data, axis=-1)
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
    df_chunk['rms_ap'] = ibldsp.utils.rms(data, axis=-1)
    return df_chunk


def dart_subtraction_numpy(data, fs, geometry, **params):
    """
    :param data: [nc, ns] numpy array of voltage traces, z-scored or not
    :return:
    """

    params = DartParameters() if params is None else DartParameters(**params)
    # pip install ephys-atlas[gpu]
    import dartsort  # 04a23714d77f28c1bbf3351ed9e21601395d1bca is a working commit
    import spikeinterface.core as sc
    import h5py

    dart_xy = np.c_[geometry['x'], geometry['y']]

    zdata = data / ibldsp.utils.rms(data, axis=-1)[:, np.newaxis]
    rec_np = sc.NumpyRecording(zdata.T, sampling_frequency=fs)
    rec_np.set_dummy_probe_from_locations(dart_xy)

    # I'm making configuration objects here that don't require fitting any
    # models. For instance, if you have do_tpca_denoise=True, dartsort will try
    # to load up many waveforms from the recording to fit a PCA, but the recording
    # is too short for that and it takes time.
    denoising_cfg = dartsort.FeaturizationConfig(
        denoise_only=True,
        do_tpca_denoise=False,
        localization_radius=params.localization_radius,
    )
    subtraction_cfg = dartsort.SubtractionConfig(
        subtraction_denoising_config=denoising_cfg,
        extract_radius=params.localization_radius,
        chunk_length_samples=params.chunk_length_samples,
    )
    # this determines what features you get out at the end
    # the nn localizer is another model which needs to be fitted, so turning
    # that off is good
    featurization_cfg = dartsort.FeaturizationConfig(
        nn_localization=False,
        save_output_waveforms=True,  # save final nn denoised waveforms
        save_input_waveforms=True,  # save collision-cleaned, but not NN-denoised, waveforms
        localization_radius=params.localization_radius,
    )

    # we make sure that each runs get a different temp folder
    temp_suffix = ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(8)])
    detected_spikes, h5_filename = dartsort.subtract(
        rec_np,
        temp_folder := Path.home().joinpath('scratch', f"dart_{temp_suffix}"),
        featurization_config=featurization_cfg,
        subtraction_config=subtraction_cfg,
        n_jobs=0,
        # if you set n_jobs=1, this will initialize CUDA in a separate process, so GPU memory will be freed. with n_jobs=0, the cuda runtime will be initialized in the main process
        show_progress=True,
    )

    df_spikes = pd.DataFrame({
        'sample': detected_spikes.times_samples,
        'channel': detected_spikes.channels,
        'ptp': detected_spikes.denoised_ptp_amplitudes,
        'xloc': detected_spikes.point_source_localizations[:, 0],  # xyza
        'yloc': detected_spikes.point_source_localizations[:, 1],  # xyza
        'zloc': detected_spikes.point_source_localizations[:, 2],  # xyza
        'alpha': detected_spikes.point_source_localizations[:, 3],  # xyza

    })

    h5file = h5py.File(h5_filename)
    d_waveforms = { # n_spikes, nsw, ncw
        'raw': np.array(h5file['collisioncleaned_waveforms']),
        'denoised': np.array(h5file['denoised_waveforms']),
        'channel_index':  np.array(h5file['channel_index'])
    }
    shutil.rmtree(temp_folder)
    return df_spikes, d_waveforms


def spikes(data, fs: int, geometry: dict, **params):
    """
    :param data:
    :param fs:
    :param geometry:
    :param params:
    :return:
    """
    params = DartParameters() if params is None else DartParameters(**params)
    df_spikes_, d_waveforms = dart_subtraction_numpy(data, fs, geometry, params=params)
    df_waveforms = ibldsp.waveforms.compute_spike_features( d_waveforms['denoised'])
    df_spikes = df_spikes_.merge(df_waveforms, left_index=True, right_index=True)
    fcn_mean_time = lambda x: np.mean((x - params.trough_offset)) / fs
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
