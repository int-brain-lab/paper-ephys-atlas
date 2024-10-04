from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import matplotlib.pyplot as plt
from iblatlas.atlas import BrainRegions
import numpy as np
from ibldsp.waveforms import double_wiggle
from pathlib import Path
from ephys_atlas.encoding import FEATURES_LIST
from ephys_atlas.plots import color_map_feature
from ibldsp import waveforms

one = ONE()
regions = BrainRegions()
color_set = color_map_feature(feature_list=FEATURES_LIST, default=False,
                              return_dict=True)

folder_file_save = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/Fig1')

pids = ['d7ec0892-0a6c-4f4f-9d8f-72083692af5c',
        '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
        'eebcaf65-7fa4-4118-869d-a084e84530e2']

pid = pids[2]
##
# Instantiate spike sorting loader
ssl = SpikeSortingLoader(pid=pid, one=one)
# Get waveforms
wavs = ssl.load_spike_sorting_object('waveforms')

# Get spikes
spikes, clusters, channels = ssl.load_spike_sorting()

##
# Plot waveform shape across channels

# IDs of wavs
idx_wavs = np.array([423, 59, 43])
fig, axs = plt.subplots(1, len(idx_wavs))

for inc in range(0, len(idx_wavs)):
    wavi = idx_wavs[inc]
    acronym = channels.acronym[clusters.channels[wavi]]

    double_wiggle(wavs['templates'][wavi] * 1e6 / 80, fs=sr_ap.fs, ax=axs[inc])
    axs[inc].set_title(f'{acronym} - Chan.{clusters.channels[wavi]}')

fig.set_size_inches([16.4, 4.8])
# Save figure
plt.savefig(folder_file_save.joinpath(f"wavs{idx_wavs}_{pid}.svg"))
plt.savefig(folder_file_save.joinpath(f"wavs{idx_wavs}_{pid}.pdf"),
            format="pdf", bbox_inches="tight")
plt.show()

##
# Plot peak tip trough on peak channel

# wavs['template'] shape is (wav, time, trace), but need arr_in dimensions: (wav, time, trace)
# Swap axis 1-2
wavs_swap = np.swapaxes(wavs['templates'], 1, 2)
arr_in = wavs_swap[idx_wavs, :, :]
# Compute features (e.g. peak, tip, trough)
df = waveforms.compute_spike_features(arr_in)
# Plot
# ----- Plot waveforms -----
fig, axs = plt.subplots(1, arr_in.shape[0])
for nth_wav in range(0, arr_in.shape[0]):
    waveforms.plot_peaktiptrough(df, arr_in, axs[nth_wav], nth_wav=nth_wav, plot_grey=False)
    axs[nth_wav].title.set_text(f'Unit #{idx_wavs[nth_wav]}')

# Add main title and set size format
fig.suptitle(f'{pid}')
fig.tight_layout()
fig.set_size_inches([11.77, 3.21])
# Save and close figure
plt.savefig(folder_file_save.joinpath(f'wavs_features_{pid}_{str(idx_wavs)}.svg'))
plt.close()