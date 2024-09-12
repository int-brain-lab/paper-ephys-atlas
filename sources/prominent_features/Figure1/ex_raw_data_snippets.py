##
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from brainbox.ephys_plots import plot_brain_regions
from ibllib.plots.misc import Density
import matplotlib.pyplot as plt
from ibldsp.voltage import destripe
from iblatlas.atlas import BrainRegions

one = ONE()
regions = BrainRegions()

# pid for a good insertion
# pid = 'dab512bd-a02d-4c1f-8dbc-9155a163efc0'
# http://reveal.internationalbrainlab.org.s3-website-us-east-1.amazonaws.com/benchmarks.html

pids = ['d7ec0892-0a6c-4f4f-9d8f-72083692af5c', '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
        'eebcaf65-7fa4-4118-869d-a084e84530e2'] #'6638cfb3-3831-4fc2-9327-194b76cf22e1',
##
for pid in pids:
    # Instantiate spike sorting loader
    ssl = SpikeSortingLoader(pid=pid, one=one)

    # Get AP spikeglx.Reader objects
    sr_ap = ssl.raw_electrophysiology(band="ap")

    # Load the raw data snippet and destripe it
    t0 = 100  # Seconds in the recording
    s0 = int(sr_ap.fs * t0)
    dur = int(0.09 * sr_ap.fs)  # We take 0.09 second of data
    raw_ap = sr_ap[s0:s0 + dur, :-sr_ap.nsync].T
    destriped = destripe(raw_ap, sr_ap.fs)
    destriped_trunc = destriped[:, 1100:int(1100+20/1000*sr_ap.fs)]

    # Get channels
    channels = ssl.load_channels()

    # Display using Density
    fig, axs = plt.subplots(1, 2, gridspec_kw={
        'width_ratios': [.95, .05]}, figsize=(8, 9), sharex='col')
    d = Density(destriped_trunc, fs=sr_ap.fs, taxis=1, gain=-91., ax=axs[0])
    # Plot brain regions in another column
    plot_brain_regions(channels["atlas_id"], channel_depths=channels["axial_um"], ax = axs[1], display=True)
    plt.show()
