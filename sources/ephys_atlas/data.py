import logging
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

import neuropixel
from one.remote import aws
from iblutil.numerical import ismember

from ibllib.atlas import Insertion
from ibllib.atlas import NeedlesAtlas, AllenAtlas
from ibllib.pipes.histology import interpolate_along_track


_logger = logging.getLogger('ibllib')

SPIKES_ATTRIBUTES = ['clusters', 'times', 'depths', 'amps']
CLUSTERS_ATTRIBUTES = ['channels', 'depths', 'metrics']

EXTRACT_RADIUS_UM = 200  # for localisation , the default extraction radius in um


def get_waveforms_coordinates(trace_indices, xy=None, extract_radius_um=EXTRACT_RADIUS_UM, return_complex=False):
    """
    Args:
        trace_indices:
        xy:
        extract_radius_um:
        return_indices:

    Returns:
    """
    if xy is None:
        th = neuropixel.trace_header(version=1)
        xy = th['x'] + 1j * th['y']
    channel_lookups = _get_channel_distances_indices(xy, extract_radius_um=extract_radius_um)
    inds = channel_lookups[trace_indices.astype(np.int32)]
    # add a dummy channel to have nans in the coordinates
    inds[np.isnan(inds)] = xy.size
    wxy = np.r_[xy, np.nan][inds.astype(np.int32)]
    if return_complex:
        return wxy
    else:
        return np.stack((np.real(wxy), np.imag(wxy), np.zeros_like(np.imag(wxy))), axis=2)


def _get_channel_distances_indices(xy, extract_radius_um=EXTRACT_RADIUS_UM):
    """
    params: xy: ntr complex array of x and y coordinates of each channel relative to the probe
    Computes the distance between each channel and all the other channels, and find the
    indices of the channels that are within the radius.
    For each row the indices of the channels within the radius are returned.
    returns: channel_dist: ntr x ntr_wav matrix of channel indices within the radius., where ntr_wav is the
    """
    ntr = xy.shape[0]
    channel_dist = np.zeros((ntr, ntr)) * np.nan
    for i in np.arange(ntr):
        cind = np.where(np.abs(xy[i] - xy) <= extract_radius_um)[0]
        channel_dist[i, :cind.size] = cind
    # prune the matrix: only so many channels are within the radius
    channel_dist = channel_dist[:, ~np.all(np.isnan(channel_dist), axis=0)]
    return channel_dist


def atlas_pids(one, tracing=True):
    django_strg = [
        'session__project__name__icontains,ibl_neuropixel_brainwide_01',
        'session__qc__lt,50',
        '~json__qc,CRITICAL',
        # 'session__extended_qc__behavior,1',
        'session__json__IS_MOCK,False',
    ]
    if tracing:
        django_strg.append('json__extended_qc__tracing_exists,True')

    insertions = one.alyx.rest('insertions', 'list', django=django_strg)
    return [item['id'] for item in insertions], insertions


def load_tables(local_path, verify=True):
    local_path.mkdir(exist_ok=True)  # no parent here
    df_clusters = pd.read_parquet(local_path.joinpath('clusters.pqt'))
    df_channels = pd.read_parquet(local_path.joinpath('channels.pqt'))
    df_voltage = pd.read_parquet(local_path.joinpath('raw_ephys_features.pqt'))
    df_probes = pd.read_parquet(local_path.joinpath('probes.pqt'))
    if verify:
        verify_tables(df_voltage, df_clusters, df_channels)
    return df_voltage, df_clusters, df_channels, df_probes


def download_tables(local_path, label='2022_W34', one=None, verify=True):
    # The AWS private credentials are stored in Alyx, so that only one authentication is required
    local_path = Path(local_path).joinpath(label)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    local_files = aws.s3_download_folder(f"aggregates/atlas/{label}", local_path, s3=s3, bucket_name=bucket_name)
    print(local_files)
    return load_tables(local_path=local_path, verify=verify)


def verify_tables(df_voltage, df_clusters, df_channels):
    """
    Verify that the tables have the correct format and indices
    :param df_clusters:
    :param df_channels:
    :param df_voltage:
    :return:
    """
    assert df_clusters.index.names == ['pid', 'cluster_id']
    assert df_channels.index.names == ['pid', 'channel']
    assert df_voltage.index.names == ['pid', 'channel']


def compute_depth_dataframe(df_raw_features, df_clusters, df_channels):
    """
    Compute a features dataframe for each pid and depth along the probe,
    merging the raw voltage features, and the clusters features
    :param df_voltage:
    :param df_clusters:
    :param df_channels:
    :return:
    """
    df_depth_clusters = df_clusters.groupby(['pid', 'axial_um']).agg(
        amp_max=pd.NamedAgg(column="amp_max", aggfunc="mean"),
        amp_min=pd.NamedAgg(column="amp_min", aggfunc="mean"),
        amp_median=pd.NamedAgg(column="amp_median", aggfunc="mean"),
        amp_std_dB=pd.NamedAgg(column="amp_std_dB", aggfunc="mean"),
        contamination=pd.NamedAgg(column="contamination", aggfunc="mean"),
        contamination_alt=pd.NamedAgg(column="contamination_alt", aggfunc="mean"),
        drift=pd.NamedAgg(column="drift", aggfunc="mean"),
        missed_spikes_est=pd.NamedAgg(column="missed_spikes_est", aggfunc="mean"),
        noise_cutoff=pd.NamedAgg(column="noise_cutoff", aggfunc="mean"),
        presence_ratio=pd.NamedAgg(column="presence_ratio", aggfunc="mean"),
        presence_ratio_std=pd.NamedAgg(column="presence_ratio_std", aggfunc="mean"),
        slidingRP_viol=pd.NamedAgg(column="slidingRP_viol", aggfunc="mean"),
        spike_count=pd.NamedAgg(column="spike_count", aggfunc="mean"),
        firing_rate=pd.NamedAgg(column="firing_rate", aggfunc="mean"),
        label=pd.NamedAgg(column="label", aggfunc="mean"),
        x=pd.NamedAgg(column="x", aggfunc="mean"),
        y=pd.NamedAgg(column="y", aggfunc="mean"),
        z=pd.NamedAgg(column="z", aggfunc="mean"),
        acronym=pd.NamedAgg(column="acronym", aggfunc="first"),
        atlas_id=pd.NamedAgg(column="atlas_id", aggfunc="first"),
    )

    df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)
    df_depth_raw = df_voltage.groupby(['pid', 'axial_um']).agg(
        alpha_mean=pd.NamedAgg(column="alpha_mean", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha_std", aggfunc="mean"),
        spike_count=pd.NamedAgg(column="spike_count", aggfunc="mean"),
        cloud_x_std=pd.NamedAgg(column="cloud_x_std", aggfunc="mean"),
        cloud_y_std=pd.NamedAgg(column="cloud_y_std", aggfunc="mean"),
        cloud_z_std=pd.NamedAgg(column="cloud_z_std", aggfunc="mean"),
        peak_trace_idx=pd.NamedAgg(column="peak_trace_idx", aggfunc="mean"),
        peak_time_idx=pd.NamedAgg(column="peak_time_idx", aggfunc="mean"),
        peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
        trough_time_idx=pd.NamedAgg(column="trough_time_idx", aggfunc="mean"),
        trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
        tip_time_idx=pd.NamedAgg(column="tip_time_idx", aggfunc="mean"),
        tip_val=pd.NamedAgg(column="tip_val", aggfunc="mean"),
        rms_ap=pd.NamedAgg(column="rms_ap", aggfunc="mean"),
        rms_lf=pd.NamedAgg(column="rms_lf", aggfunc="mean"),
        psd_delta=pd.NamedAgg(column="psd_delta", aggfunc="mean"),
        psd_theta=pd.NamedAgg(column="psd_theta", aggfunc="mean"),
        psd_alpha=pd.NamedAgg(column="psd_alpha", aggfunc="mean"),
        psd_beta=pd.NamedAgg(column="psd_beta", aggfunc="mean"),
        psd_gamma=pd.NamedAgg(column="psd_gamma", aggfunc="mean"),
        x=pd.NamedAgg(column="x", aggfunc="mean"),
        y=pd.NamedAgg(column="y", aggfunc="mean"),
        z=pd.NamedAgg(column="z", aggfunc="mean"),
        acronym=pd.NamedAgg(column="acronym", aggfunc="first"),
        atlas_id=pd.NamedAgg(column="atlas_id", aggfunc="first"),
        histology=pd.NamedAgg(column="histology", aggfunc="first"),
    )
    df_depth = df_depth_raw.merge(df_depth_clusters, left_index=True, right_index=True)
    return df_depth


def compute_channels_micromanipulator_coordinates(df_channels, one=None):
    """
    Compute the micromanipulator coordinates for each channel
    :param df_channels:
    :param one:
    :return:
    """
    assert one is not None, 'one instance must be provided to fetch the planned trajectories'
    needles = NeedlesAtlas()
    allen = AllenAtlas()
    needles.compute_surface()

    pids = df_channels.index.levels[0]
    trajs = one.alyx.rest('trajectories', 'list', provenance='Micro-Manipulator')

    mapping = dict(
        pid='probe_insertion', pname='probe_name', x_target='x', y_target='y', z_target='z',
        depth_target='depth', theta_target='theta', phi_target='phi', roll_target='roll'
    )

    tt = [{k: t[v] for k, v in mapping.items()} for t in trajs]
    df_planned = pd.DataFrame(tt).rename(columns={'probe_insertion': 'pid'}).set_index('pid')
    df_planned['eid'] = [t['session']['id'] for t in trajs]

    df_probes = df_channels.groupby('pid').agg(histology=pd.NamedAgg(column='histology', aggfunc='first'))
    df_probes = pd.merge(df_probes, df_planned, how='left', on='pid')

    iprobes, iplan = ismember(pids, df_planned.index)
    # imiss = np.where(~iprobes)[0]

    for pid, rec in df_probes.iterrows():
        drec = rec.to_dict()
        ins = Insertion.from_dict({v: drec[k] for k, v in mapping.items() if 'target' in k}, brain_atlas=needles)
        txyz = np.flipud(ins.xyz)
        txyz = allen.bc.i2xyz(needles.bc.xyz2i(txyz / 1e6, round=False, mode="clip")) * 1e6
        # we interploate the channels from the deepest point up. The neuropixel y coordinate is from the bottom of the probe
        xyz_mm = interpolate_along_track(txyz, df_channels.loc[pid, 'axial_um'].to_numpy() / 1e6)
        aid_mm = needles.get_labels(xyz=xyz_mm, mode='clip')
        df_channels.loc[pid, 'x_target'] = xyz_mm[:, 0]
        df_channels.loc[pid, 'y_target'] = xyz_mm[:, 1]
        df_channels.loc[pid, 'z_target'] = xyz_mm[:, 2]
        df_channels.loc[pid, 'atlas_id_target'] = aid_mm
    return df_channels, df_probes


def get_config():
    file_yaml = Path(__file__).parents[2].joinpath('config-ephys-atlas.yaml')
    with open(file_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
    return config
