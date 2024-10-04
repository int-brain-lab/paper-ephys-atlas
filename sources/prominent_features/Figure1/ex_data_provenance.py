##
from nptdms.tdmsinfo import display
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from brainbox.ephys_plots import plot_brain_regions
from ibllib.plots.misc import Density
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ibldsp.voltage import destripe, destripe_lfp
from iblatlas.atlas import BrainRegions
import numpy as np
from ibldsp.waveforms import double_wiggle
from pathlib import Path
import scipy.signal
from ephys_atlas.encoding import FEATURES_LIST
from ephys_atlas.plots import color_map_feature
from ephys_atlas.features import BANDS

one = ONE()
regions = BrainRegions()
color_set = color_map_feature(feature_list=FEATURES_LIST, default=False,
                              return_dict=True)

folder_file_save = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/Fig1')

# http://reveal.internationalbrainlab.org.s3-website-us-east-1.amazonaws.com/benchmarks.html

pids = ['d7ec0892-0a6c-4f4f-9d8f-72083692af5c',
        '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
        'eebcaf65-7fa4-4118-869d-a084e84530e2']

pid = pids[2]

# Define color palette for bands
COLOR_BANDS = {'delta': '#DEEFB7', 'theta': '#98DFAF', 'alpha': '#5FB49C', 'beta': '#CDFCE4', 'gamma': '#88D5CC'}

##
# LFP function to compute and plot PSD
def show_psd_1trace(data, fs, ax=None, f_lim=None, bands=BANDS, ylim=(-140, -90), display='log'):
    ''' Computes and shows PSD for 1 trace '''
    # Compute PSD
    f, psd = scipy.signal.welch(data, fs=fs, nperseg=512)

    # Restrict freq range for plotting, e.g. to 0-250Hz
    if f_lim is not None:
        indx_f = np.where(f < f_lim)[0]
        f = f[indx_f]
        psd = psd[indx_f]

    # Plot
    if ax is None:
        fig, ax = plt.subplots()

    if display == 'log':
        ax.set_xscale('log')
        ax.semilogx(f, 10 * np.log10(psd), color='black')

    elif display == 'linear':
        ax.plot(f, 10 * np.log10(psd), color='black', alpha=0.5)
        if f_lim is None:
            ax.set_xlim(0, fs / 2)
        else:
            ax.set_xlim(0, f_lim)

    if bands is not None:
        # Plot rectangle over PSD
        for key, value in bands.items():
            if key not in ['lfp']:  # Take all bands for plotting but LFP
                xy = (bands[key][0], ylim[0])
                width = bands[key][1] - bands[key][0]
                height = ylim[1] - ylim[0]
                ax.add_patch(Rectangle(xy, width, height, color=COLOR_BANDS[key]))

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB rel V/Hz)')
    ax.set_ylim(ylim)


##
# Instantiate spike sorting loader
ssl = SpikeSortingLoader(pid=pid, one=one)

# Get AP / LF spikeglx.Reader objects
sr_ap = ssl.raw_electrophysiology(band="ap")
sr_lf = ssl.raw_electrophysiology(band="lf")

# Load the raw data snippet and destripe it
t0 = 100  # Seconds in the recording
s0 = int(sr_ap.fs * t0)
dur = int(1 * sr_ap.fs)  # We take 1 second of data
ms_start_display = 500
ms_dur_display = 400

# Get increments for AP/LF and load data
inc_start_ap = int(ms_start_display / 1000 * sr_ap.fs)
inc_dur_ap = int(ms_dur_display / 1000 * sr_ap.fs)
raw_ap = sr_ap[s0:s0 + dur, :-sr_ap.nsync].T
destriped_ap = destripe(raw_ap, sr_ap.fs)
destriped_ap_trunc = destriped_ap[:, inc_start_ap:inc_start_ap + inc_dur_ap]

inc_start_lf = int(ms_start_display / 1000 * sr_lf.fs)
inc_dur_lf = int(ms_dur_display / 1000 * sr_lf.fs)
raw_lf = sr_lf[s0:s0 + dur, :-sr_ap.nsync].T

destriped_lf = destripe_lfp(raw_lf, sr_lf.fs)
destriped_lf_trunc = destriped_lf[:, inc_start_lf:inc_start_lf + inc_dur_lf]

# Get waveforms
wfs = ssl.load_spike_sorting_object('waveforms')

# Get channels
# channels = ssl.load_channels()

# Get spikes, especially the samples to align it with the raw data
spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=["spikes.samples"])

# Get the spikes that are within this raw data snippet
spike_selection = slice(*np.searchsorted(spikes.samples, [s0+inc_start_ap, s0+inc_start_ap + inc_dur_ap]))
su = spikes.clusters[spike_selection]
sc = clusters.channels[spikes.clusters][spike_selection]
ss = (spikes.samples[spike_selection] - s0) / sr_ap.fs * 1e3

##
# Display using Density
# (This is not used in the Fig, just to visualise and create the further plots)

fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [.95, .05]}, figsize=(18, 12))
Density(destriped_ap_trunc, fs=sr_ap.fs, taxis=1, ax=axs[0, 0])
plot_brain_regions(channels["atlas_id"], channel_depths=channels["axial_um"], ax = axs[0, 1], display=True)
Density(- destriped_lf_trunc, fs=sr_lf.fs, taxis=1, ax=axs[1, 0])
plot_brain_regions(channels["atlas_id"], channel_depths=channels["axial_um"], ax = axs[1, 1], display=True)

plt.show()

##
# Set y-lim
ylim = (-0.0002, 0.0003)
# Select channel to display in 1D
ch_ids = [30, 350]

for ch_id in ch_ids:
    ch_acronym = channels['acronym'][ch_id]

    fig, axs = plt.subplots(1,3)
    # PSD
    # For this we take the whole streamed signal (1s data)  #TODO remove borders for filter effect?
    show_psd_1trace(destriped_lf[ch_id, :], fs=sr_lf.fs, ax=axs[0], f_lim=250)
    axs[0].set_title(ch_acronym)

    # LF
    ax = axs[1]
    ch_display = destriped_lf_trunc[ch_id, :]
    ax.plot(ch_display, color=color_set['raw_lf'])
    ax.set_ylim(ylim)

    #AP
    ax = axs[2]
    ch_display = destriped_ap_trunc[ch_id, :]
    ax.plot(ch_display, color=color_set['raw_ap'])
    ax.set_ylim(ylim)

    fig.set_size_inches([16.4, 4.8])
    fig.tight_layout()

    # Save figure
    plt.savefig(folder_file_save.joinpath(f"AP_LFP_PSD__ch{ch_id}_T0{t0}_start"
                                          f"{ms_start_display}_dur{ms_dur_display}_{pid}.svg"))

##
wfs = ssl.load_spike_sorting_object('waveforms')
wfs['templates'].shape

# IDs of wavs
wavs = [423, 59, 43]
fig, axs = plt.subplots(1, len(wavs))

for inc in range(0, len(wavs)):
    wavi = wavs[inc]
    acronym = channels.acronym[clusters.channels[wavi]]

    double_wiggle(wfs['templates'][wavi] * 1e6 / 80, fs=sr_ap.fs, ax=axs[inc])
    axs[inc].set_title(f'{acronym}')

fig.set_size_inches([16.4, 4.8])
# Save figure
plt.savefig(folder_file_save.joinpath(f"wavs{wavs}_{pid}.svg"))
plt.savefig(folder_file_save.joinpath(f"wavs{wavs}_{pid}.pdf"),
            format="pdf", bbox_inches="tight")
plt.show()
