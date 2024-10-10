"""
Fig
Brain regions and coronal slice for selected Traj

"""

from one.api import ONE
from iblatlas import atlas
import numpy as np
from ibllib.pipes.ephys_alignment import EphysAlignment
from pathlib import Path
import matplotlib.pyplot as plt
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_voltage_features

# Instantiate brain atlas and one
one = ONE()
brain_atlas = atlas.AllenAtlas(25)
br = BrainRegions()

label = 'latest'
mapping = 'Allen'

local_data_path = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/Data')
save_folder = Path('/Users/gaellechapuis/Desktop/Reports/EphysAtlas/Test')

if not save_folder.parent.exists():
    save_folder.parent.mkdir()
if not save_folder.exists():
    save_folder.mkdir()

df_voltage, df_clusters, df_channels, df_probes = \
    load_voltage_features(local_data_path.joinpath(label), mapping=mapping)
# Do not remove void / root
##
# Prepare the dataframe for a single probe
# pid = '0ee04753-3039-4209-bed8-5c60e38fe5da'
# pid = '0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a'
# pid = 'f362c84f-8d9a-4d5b-8439-055ae936fdff'
pid = 'eebcaf65-7fa4-4118-869d-a084e84530e2'
pid_ch_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

# Create numpy array of xyz um
xyz_channels = pid_ch_df[['x', 'y', 'z']].to_numpy()
# xyz_channels = pid_ch_df[['x_target', 'y_target', 'z_target']].to_numpy()
depths = pid_ch_df[['axial_um']].to_numpy().squeeze()
provenance = pid_ch_df['histology'].unique()[0]
##
# Get image, inpired from:
# https://int-brain-lab.github.io/iblenv/notebooks_external/docs_find_previous_alignments.html

xyz_picks = xyz_channels
ephysalign = EphysAlignment(xyz_picks, depths)

# Find brain region that each channel is located in
brain_regions = ephysalign.get_brain_locations(xyz_channels)

# For plotting -> extract the boundaries of the brain regions, as well as CCF label and colour
region, region_label, region_colour, _ = ephysalign.get_histology_regions(xyz_channels, depths)


# Create a figure and arrange using gridspec
widths = [1, 2.5]
heights = [1] * 1
gs_kw = dict(width_ratios=widths, height_ratios=heights)
fig, axis = plt.subplots(1, 2, constrained_layout=True,
                         gridspec_kw=gs_kw, figsize=(8, 9))

# Make plot that shows the brain regions that channels pass through
ax_regions = fig.axes[0]
for reg, col in zip(region, region_colour):
    height = np.abs(reg[1] - reg[0])
    bottom = reg[0]
    color = col / 255
    ax_regions.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')
ax_regions.set_yticks(region_label[:, 0].astype(int))
ax_regions.yaxis.set_tick_params(labelsize=8)
ax_regions.get_xaxis().set_visible(False)
ax_regions.set_yticklabels(region_label[:, 1])
ax_regions.spines['right'].set_visible(False)
ax_regions.spines['top'].set_visible(False)
ax_regions.spines['bottom'].set_visible(False)
ax_regions.hlines([0, 3840], *ax_regions.get_xlim(), linestyles='dashed', linewidth=3,
                  colors='k')
# ax_regions.plot(np.ones(channel_depths_track.shape), channel_depths_track, '*r')

# Make plot that shows coronal slice that trajectory passes through with location of channels
# shown in black
ax_slice = fig.axes[1]
brain_atlas.plot_tilted_slice(xyz_channels, axis=1, ax=ax_slice, volume='annotation')
ax_slice.plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, 'k*')
ax_slice.title.set_text(f'pid {pid} - {provenance}')


# Make sure the plot displays
# plt.show()

# save
plt.savefig(save_folder.joinpath(f'{pid}_{provenance}_traj_from_df.svg'))
