from pathlib import Path
from typing import Union
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from ibllib.atlas import BrainRegions
from brainbox.io.one import SpikeSortingLoader
import spikeglx
from one.api import ONE
from one.remote import aws
from ibllib.plots import wiggle
from neurodsp.utils import rms
from viewephys.gui import viewephys, EphysViewer


class AtlasDataModel(object):

    def __init__(self, ROOT_PATH, one, pid, t0=2500):
        self.ROOT_PATH = ROOT_PATH
        self.one = one
        self.pid = pid
        self.regions = BrainRegions()
        self.path_pid = self.ROOT_PATH.joinpath(self.pid)
        self.path_qc = self.path_pid.joinpath('pics')
        self.T0 = t0
        path_t0 = next(self.path_pid.glob(f'T0{self.T0}*'))

        self.path_qc.mkdir(exist_ok=True)
        self.ap = np.load(path_t0.joinpath('ap.npy')).astype(np.float32)
        self.raw = np.load(path_t0.joinpath('raw.npy')).astype(np.float32)
        self.zscore = rms(self.ap)
        with open(path_t0.joinpath('ap.yml'), 'r') as f:
            ap_info = yaml.safe_load(f)
        self.fs_ap = ap_info['fs']

        self.spikes = pd.read_parquet(path_t0.joinpath('spikes.pqt')) if path_t0.joinpath('spikes.pqt').exists() else None
        self.waveforms = np.load(path_t0.joinpath('waveforms.npy'))

        ssl = SpikeSortingLoader(pid=pid, one=one)
        self.channels = ssl.load_channels()
        self.eqcs = {}

    def view(self, alpha_min=100):

        self.eqcs['ap'] = viewephys(self.ap, fs=self.fs_ap, title='ap', channels=self.channels, br=self.regions)
        sel = self.spikes['alpha'] > alpha_min
        self.eqcs['ap'].ctrl.add_scatter(
            self.spikes['sample'][sel] / self.fs_ap * 1e3,
            self.spikes['trace'][sel],
            (0, 255, 0, 100),
            label='spikes')


        sl = self.eqcs['ap'].layers['spikes']
        try:
            sl['layer'].sigClicked.disconnect()
        except:
            pass
        sl['layer'].sigClicked.connect(self.click_on_spike_callback)

    def click_on_spike_callback(self, obj, toto, event):
        t = event.pos().x() / 1e3
        c = int(event.pos().y())
        fs = self.fs_ap
        ispi = np.arange(
            np.searchsorted(self.spikes['sample'], int(t * fs) - 5),
            np.searchsorted(self.spikes['sample'], int(t * fs) + 5) + 1
        )
        iw = ispi[np.argmin(np.abs(self.spikes['trace'].iloc[ispi] - c))]
        print(iw)

        rwav, hwav, cind, sind = self.getwaveform(iw, return_indices=True)
        wav = np.squeeze(self.waveforms[iw, :, :] * self.zscore[cind])

        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        wiggle(- wav, fs=fs, gain=40, ax=axs[0])
        wiggle(- rwav.T, fs=fs, gain=40, ax=axs[1])
        wiggle(- rwav.T + wav, fs=fs, gain=40, ax=axs[2])
        # sns.histplot(spikes['alpha'])

        # import scipy.signal
        # sos = scipy.signal.butter(N=3, Wn=6000 / 30000 * 2, btype='lowpass', output='sos')
        # lowpass = scipy.signal.sosfiltfilt(sos, ap)
        # eqcs['lowpass'] = viewephys(lowpass, fs=fs, title='lowpass', channels=channels, br=regions)


    # def download_data(self, pid):
    #     path_pid = ROOT_PATH.joinpath(pid)
    #     if not path_pid.exists():
    #         s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    #         aws.s3_download_folder(f"resources/ephys-atlas-sample/{pid}", ROOT_PATH.joinpath(pid), s3=s3,
    #                                bucket_name=bucket_name)
    @property
    def xy(self):
        return self.channels['lateral_um'] + 1j * self.channels['axial_um']

    def getwaveform(self, iw, extract_radius=200, trough_offset=42, spike_length_samples=121, return_indices=False):
        s0 = int(self.spikes['sample'].iloc[iw] - trough_offset)
        sind = slice(s0, s0 + int(spike_length_samples))

        cind = np.abs(self.xy[int(self.spikes['trace'].iloc[iw])] - self.xy) <= extract_radius
        hwav = {k: v[cind] for k, v in self.channels.items()}
        if return_indices:
            return self.ap[cind, sind], hwav, cind, sind
        else:
            return self.ap[cind, sind], hwav
