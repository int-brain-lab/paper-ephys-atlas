import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neurodsp.waveforms import plot_peaktiptrough, compute_spike_features


from ibllib.atlas import BrainRegions
from brainbox.io.one import SpikeSortingLoader
from ibllib.plots import wiggle
from neurodsp.utils import rms
from viewephys.gui import viewephys
from neurodsp.voltage import kfilt


class AtlasDataModel(object):

    def __init__(self, ROOT_PATH, one, pid, t0=1500):
        """
        :param ROOT_PATH:
        :param one:
        :param pid:
        :param t0:
        """
        self.ROOT_PATH = ROOT_PATH
        self.one = one
        self.pid = pid
        self.regions = BrainRegions()
        self.path_pid = self.ROOT_PATH.joinpath(self.pid)
        self.path_qc = self.path_pid.joinpath('pics')
        self.T0 = t0
        path_t0 = next(self.path_pid.glob(f'T{str(self.T0).zfill(5)}*'))

        self.path_qc.mkdir(exist_ok=True)
        if path_t0.joinpath('destriped.npy').exists():
            self.ap = np.load(path_t0.joinpath('destriped.npy')).astype(np.float32)
        else:
            self.ap = np.load(path_t0.joinpath('ap.npy'))
        self.ap_raw = np.load(path_t0.joinpath('ap_raw.npy')).astype(np.float32)
        self.zscore = rms(self.ap)
        with open(path_t0.joinpath('ap.yml'), 'r') as f:
            ap_info = yaml.safe_load(f)
        self.fs_ap = ap_info['fs']

        self.spikes = pd.read_parquet(path_t0.joinpath('spikes.pqt')) if path_t0.joinpath('spikes.pqt').exists() else None
        self.waveforms = np.load(path_t0.joinpath('waveforms.npy'))

        if path_t0.joinpath('channels.npy').exists():
            ch_load = np.load(path_t0.joinpath('channels.npy'), allow_pickle=True)
            self.channels = dict(ch_load.item())
        else:
            ssl = SpikeSortingLoader(pid=pid, one=one)
            self.channels = ssl.load_channels()
        self.eqcs = {}
        self.kfilt = None

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

    def view_kfilt(self):
        if self.kfilt is None:
            self.kfilt = kfilt(self.ap)
        self.eqcs['kfilt'] = viewephys(self.kfilt, fs=self.fs_ap, title='kfilt', channels=self.channels, br=self.regions)

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

        fig, axs = plt.subplots(2, 3, sharex='row', sharey='row')
        wiggle(- wav, fs=fs, gain=40, ax=axs[0, 0])
        wiggle(- rwav.T, fs=fs, gain=40, ax=axs[0, 1])
        wiggle(- rwav.T + wav, fs=fs, gain=40, ax=axs[0, 2])
        axs[0, 0].set_title(f'ID wav: {iw}')
        # sns.histplot(spikes['alpha'])

        # Subplot with peak-tip-trough
        # New row to remove share axis
        new_wav = wav[np.newaxis, :, :]
        df, arr_out = compute_spike_features(new_wav, return_peak_channel=True)
        plot_peaktiptrough(df, new_wav, axs[1, 0], nth_wav=0)
        axs[1, 0].set_ylabel('(Volt)')
        axs[1, 0].set_xlabel('(samples)')
        # Todo maybe better to set x-axis in (s) and fix y axis lim across spikes?
        # xlabels = axs[1, 0].get_xticklabels()
        # labels = [item.get_text() for item in xlabels]
        # new_labels = []
        # for label in labels:
        #     new_labels.append(round(float(label) / fs * 1000, 2))
        # axs[1, 0].set_xticklabels(new_labels)

        axs[1, 1].set_visible(False)
        axs[1, 2].set_visible(False)

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


