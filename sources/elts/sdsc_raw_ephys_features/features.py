import spikeglx
import pandas as pd
import scipy
import numpy as np
from tqdm import trange

from neurodsp.voltage import destripe, destripe_lfp, detect_bad_channels
from neurodsp.utils import rms
from deploy.iblsdsc import OneSdsc as ONE


def get_spikeglx_files(pid, one, band):
    """
    Get SDSC paths to cbin, meta, and ch files for an insertion (in that order).
    
    :param pid: insertion ID.
    :param one: OneSdsc instance
    :param band: "ap" or "lf".
    :return: A list of Paths to cbin, meta, and ch files (with uuids), in that order.
    """
    eid, probe = one.pid2eid(pid)
    files = []
    for suffix in ["cbin", "meta", "ch"]:
        dsets = one.list_datasets(eid=eid, collection=f"raw_ephys_data/{probe}", filename=f"*.{band}.{suffix}", details=True)
        files.append(one._check_filesystem(dsets, offline=True)[0])
    return files
    
def compute_ap_features(sr, t_start, t_dur, features_list):
    """
    Compute selected AP raw data features on snippets.
    
    :param sr: A spikeglx Reader object for the AP raw data.
    :t_start: A list of starting times in seconds.
    :t_dur: A list of snippet durations corresponding to t_start.
    :features_list: The list of AP features desired.
    :return: A Pandas dataframe.
    """
    filter = scipy.signal.butter(
        N=3, Wn=300 / sr.fs * 2, btype="highpass", output="sos"
    )
    dfs = {}
    for i in trange(len(t_start)):
        _t_start = t_start[i]
        _t_dur = min(t_dur[i], sr.ns - _t_start)
        df = compute_ap_features_snippet(sr, _t_start, _t_dur, features_list, filter)
        dfs[i] = df
        
    out_df = pd.concat(dfs)
    out_df.index.rename(["snippet_id", "channel_id"], inplace=True)

    return out_df
        
    
def compute_ap_features_snippet(sr, t_start, t_dur, features_list, filter=None):
    # conversion from V to uV
    factor = 1.0e6
    detect_kwargs = {"fs": sr.fs, "psd_hf_threshold": None}

    if filter is None:
        filter = scipy.signal.butter(
            N=3, Wn=300 / sr.fs * 2, btype="highpass", output="sos"
        )
        
    data = {}
    sl = slice(t_start, t_start + t_dur)
    raw = sr[sl, : -sr.nsync].T
    dc_offset = np.mean(raw, axis=1)
    channel_labels, xfeats_raw = detect_bad_channels(raw, **detect_kwargs)
    butter = scipy.signal.sosfiltfilt(filter, raw)
    destriped = destripe(raw, fs=sr.fs, channel_labels=channel_labels)
    # compute same channel feats for destripe
    _, xfeats_destriped = detect_bad_channels(destriped, **detect_kwargs)
    
    # get raw/destriped rms for free
    raw_rms = xfeats_raw["rms_raw"]
    destripe_rms = xfeats_destriped["rms_raw"]
    
    butter_rms = rms(butter)
    striping_rms = rms(butter - destriped)
    
    data[f"ap_dc_offset"] = dc_offset * factor
    data[f"ap_raw_rms"] = raw_rms * factor
    data[f"ap_butter_rms"] = butter_rms * factor
    data[f"ap_destripe_rms"] = destripe_rms * factor
    data[f"ap_striping_rms"] = striping_rms * factor
    data[f"ap_channel_labels"] = channel_labels
    # channel detect features
    data[f"ap_xcor_hf_raw"] = xfeats_raw["xcor_hf"]
    data[f"ap_xcor_lf_raw"] = xfeats_raw["xcor_lf"]
    data[f"ap_psd_hf_raw"] = xfeats_raw["psd_hf"]
    data[f"ap_xcor_hf_destripe"] = xfeats_destriped["xcor_hf"]
    data[f"ap_xcor_lf_destripe"] = xfeats_destriped["xcor_lf"]
    data[f"ap_psd_hf_destripe"] = xfeats_destriped["psd_hf"]
    
    return pd.DataFrame(data)
    
def compute_lf_features(sr, t_start, t_dur, features_list):
    """
    Compute selected LF raw data features on snippets.
    
    :param sr: A spikeglx Reader object for the LF raw data.
    :t_start: A list of starting times in seconds.
    :t_dur: A list of snippet durations corresponding to t_start.
    :features_list: The list of LF features desired.
    :return: A Pandas dataframe.
    """
    filter = scipy.signal.butter(
        N=3, Wn=[2 / sr.fs * 2, 200 / sr.fs * 2], btype="bandpass", output="sos"
    )
    dfs = {}
    for i in trange(len(t_start)):
        _t_start = t_start[i]
        _t_dur = min(t_dur[i], sr.ns - _t_start)
        df = compute_lf_features_snippet(sr, _t_start, _t_dur, features_list, filter)
        dfs[i] = df
        
    out_df = pd.concat(dfs)
    out_df.index.rename(["snippet_id", "channel_id"], inplace=True)

    return out_df
        
    
def compute_lf_features_snippet(sr, t_start, t_dur, features_list, filter=None):
    # conversion from V to uV
    factor = 1.0e6
    detect_kwargs = {"fs": sr.fs, "psd_hf_threshold": 1.4}

    if filter is None:
        filter = scipy.signal.butter(
            N=3, Wn=[2 / sr.fs * 2, 200 / sr.fs * 2], btype="bandpass", output="sos"
        )
        
    data = {}
    sl = slice(t_start, t_start + t_dur)
    raw = sr[sl, : -sr.nsync].T
    dc_offset = np.mean(raw, axis=1)
    channel_labels, xfeats_raw = detect_bad_channels(raw, **detect_kwargs)
    butter = scipy.signal.sosfiltfilt(filter, raw)
    destriped = destripe_lfp(raw, fs=sr.fs, channel_labels=channel_labels)
    # compute same channel feats for destripe
    _, xfeats_destriped = detect_bad_channels(destriped, **detect_kwargs)
    
    # get raw/destriped rms for free
    raw_rms = xfeats_raw["rms_raw"]
    destripe_rms = xfeats_destriped["rms_raw"]
    
    butter_rms = rms(butter)
    striping_rms = rms(butter - destriped)
    
    data[f"lf_dc_offset"] = dc_offset * factor
    data[f"lf_raw_rms"] = raw_rms * factor
    data[f"lf_butter_rms"] = butter_rms * factor
    data[f"lf_destripe_rms"] = destripe_rms * factor
    data[f"lf_striping_rms"] = striping_rms * factor
    data[f"lf_channel_labels"] = channel_labels
    # channel detect features
    data[f"lf_xcor_hf_raw"] = xfeats_raw["xcor_hf"]
    data[f"lf_xcor_lf_raw"] = xfeats_raw["xcor_lf"]
    data[f"lf_psd_hf_raw"] = xfeats_raw["psd_hf"]
    data[f"lf_xcor_hf_destripe"] = xfeats_destriped["xcor_hf"]
    data[f"lf_xcor_lf_destripe"] = xfeats_destriped["xcor_lf"]
    data[f"lf_psd_hf_destripe"] = xfeats_destriped["psd_hf"]
    
    return pd.DataFrame(data)
    

def compute_raw_features(
    pid, 
    output_dir,
    ap_t_start, 
    ap_t_dur, 
    lf_t_start, 
    lf_t_dur, 
    ap_features_list=None, 
    lf_features_list=None
):
    """
    Compute AP and LF raw features and save to a parquet file in a specified location.
    
    :param pid: insertion ID.
    :param output_dir: Location to save features table.
    :param ap_t_start: A list of starting time for AP features in seconds.
    :param ap_t_dur: A list of durations in seconds corresponding to ap_t_start.
    :param lf_t_start: A list of starting time for AP features in seconds.
    :param lf_t_dur: A list of durations in seconds corresponding to lf_t_start.
    :param ap_features_list: A list of AP features to compute.
    :param lf_features
    """
    
    one = ONE()
    
    if ap_features_list is None:
        ap_features_list = [
            "ap_dc_offset",
            "ap_raw_rms",
            "ap_butter_rms",
            "ap_destripe_rms",
            "ap_striping_rms",
            "ap_channel_labels",
            "ap_xcor_hf_raw",
            "ap_xcor_lf_raw",
            "ap_psd_hf_raw",
            "ap_xcor_hf_destripe",
            "ap_xcor_lf_destripe",
            "ap_psd_hf_destripe",
        ]
    if lf_features_list is None:
        lf_features_list = [
            "lf_dc_offset",
            "lf_raw_rms",
            "lf_butter_rms",
            "lf_destripe_rms",
            "lf_striping_rms",
            "lf_channel_labels",
            "lf_xcor_hf_raw",
            "lf_xcor_lf_raw",
            "lf_psd_hf_raw",
            "lf_xcor_hf_destripe",
            "lf_xcor_lf_destripe",
            "lf_psd_hf_destripe",
        ]
        
    ap_cbin, ap_meta, ap_ch = get_spikeglx_files(pid, one, "ap")
    lf_cbin, lf_meta, lf_ch = get_spikeglx_files(pid, one, "lf")
    
    sr_ap = spikeglx.Reader(ap_cbin, meta_file=ap_meta, ch_file=ap_ch)
    sr_lf = spikeglx.Reader(lf_cbin, meta_file=lf_meta, ch_file=lf_ch)
    
    ap_feats = compute_ap_features(sr_ap, ap_t_start, ap_t_dur, ap_features_list)
    lf_feats = compute_lf_features(sr_lf, lf_t_start, lf_t_dur, lf_features_list)
    
    # aggregate over snippets
    ap_channels = ap_feats.groupby(["channel_id"]).agg(
        ap_dc_offset = pd.NamedAgg(column="ap_dc_offset", aggfunc="median"),
        ap_raw_rms=pd.NamedAgg(column="ap_raw_rms", aggfunc="median"),
        ap_butter_rms=pd.NamedAgg(column="ap_butter_rms", aggfunc="median"),
        ap_destripe_rms=pd.NamedAgg(column="ap_destripe_rms", aggfunc="median"),
        ap_striping_rms=pd.NamedAgg(column="ap_striping_rms", aggfunc="median"),
        ap_xcor_hf_raw=pd.NamedAgg(column="ap_xcor_hf_raw", aggfunc="median"),
        ap_xcor_lf_raw=pd.NamedAgg(column="ap_xcor_lf_raw", aggfunc="median"),
        ap_psd_hf_raw=pd.NamedAgg(column="ap_psd_hf_raw", aggfunc="median"),
        ap_xcor_hf_destripe=pd.NamedAgg(column="ap_xcor_hf_destripe", aggfunc="median"),
        ap_xcor_lf_destripe=pd.NamedAgg(column="ap_xcor_lf_destripe", aggfunc="median"),
        ap_psd_hf_destripe=pd.NamedAgg(column="ap_psd_hf_destripe", aggfunc="median"),
        channel_labels=pd.NamedAgg(column="ap_channel_labels", aggfunc=lambda x: pd.Series.mode(x)[0]) 
    )
    
    lf_channels = lf_feats.groupby(["channel_id"]).agg(
        lf_dc_offset = pd.NamedAgg(column="lf_dc_offset", aggfunc="median"),
        lf_raw_rms=pd.NamedAgg(column="lf_raw_rms", aggfunc="median"),
        lf_butter_rms=pd.NamedAgg(column="lf_butter_rms", aggfunc="median"),
        lf_destripe_rms=pd.NamedAgg(column="lf_destripe_rms", aggfunc="median"),
        lf_striping_rms=pd.NamedAgg(column="lf_striping_rms", aggfunc="median"),
        lf_xcor_hf_raw=pd.NamedAgg(column="lf_xcor_hf_raw", aggfunc="median"),
        lf_xcor_lf_raw=pd.NamedAgg(column="lf_xcor_lf_raw", aggfunc="median"),
        lf_psd_hf_raw=pd.NamedAgg(column="lf_psd_hf_raw", aggfunc="median"),
        lf_xcor_hf_destripe=pd.NamedAgg(column="lf_xcor_hf_destripe", aggfunc="median"),
        lf_xcor_lf_destripe=pd.NamedAgg(column="lf_xcor_lf_destripe", aggfunc="median"),
        lf_psd_hf_destripe=pd.NamedAgg(column="lf_psd_hf_destripe", aggfunc="median"),
    )
    
    df = ap_channels.join(lf_channels)
    
    out_filename = output_dir.joinpath(f"{pid}_features.pqt")
    df.to_parquet(out_filename)