# %%
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import ephys_atlas.data
import brainbox.ephys_plots
import iblatlas.atlas
import iblatlas.plots
import matplotlib

import iblatlas.atlas
from iblatlas.gui import atlasview

ba = iblatlas.atlas.AllenAtlas()

MM_TO_INCH = 1 / 25.4

transform = {
    'polarity': lambda x: np.log10(x + 1 + 1e-6),
    'rms_lf': lambda x: 20 * np.log10(x),
    'rms_ap': lambda x: 20 * np.log10(x),
    'rms_lf_csd': lambda x: 20 * np.log10(x),
    'spike_count': lambda x: np.log10(x),
}

FIG_SIZE = (13, 6)

path_features = Path('/home/ibladmin/Downloads/2024_W50')  # parede
df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_voltage_features(path_features)

regions = iblatlas.atlas.BrainRegions()

all_pids = df_voltage.index.get_level_values(0).unique().tolist()
df_voltage['cortical_depth'] = iblatlas.atlas.xyz_to_depth(df_voltage.loc[:, ['x', 'y', 'z']].values)



# %% Plot the brain regioons for the set of pids
n_pids = 3
pids = np.random.choice(all_pids, n_pids)
widths =  [0.5, 1.5, 1] * n_pids + [1.5, 8]
fig, ax = plt.subplots(1, len(widths), figsize=FIG_SIZE, gridspec_kw={'width_ratios': widths})
ba.plot_top(ax=ax[-1])
clim = [0, 1500]

for i, pid in enumerate(pids):
    iax = i * 3
    df_pid = df_voltage.loc[pid]
    depth = df_pid.groupby('axial_um').agg(atlas_id=pd.NamedAgg(column='Allen_id', aggfunc='first')).reset_index()
    brainbox.ephys_plots.plot_brain_regions(
        depth['atlas_id'].values, channel_depths=depth['axial_um'].values, brain_regions=regions, display=True, ax=ax[iax])

    data_bank, x_bank, y_bank = brainbox.plot_base.arrange_channels2banks(
        data=df_pid['cortical_depth'].to_numpy(),
        chn_coords=df_pid[['lateral_um', 'axial_um']].to_numpy(),
        pad=True,
    )
    data = brainbox.plot_base.ProbePlot(data_bank, x=x_bank, y=y_bank, cmap='PuOr')
    data.clim = clim
    brainbox.ephys_plots.plot_probe(data.convert2dict(), ax=ax[iax + 1], show_cbar=False)

    # now displays over cortex
    ax[-1].plot(df_pid['x'].values * 1e6, df_pid['y'].values * 1e6, '.', color='r', markersize=2)
    ax[-1].plot(df_pid['x'].values[-1] * 1e6, df_pid['y'].values[-1] * 1e6, '*', color='k', markersize=2)

plt.show()


for i in np.r_[np.arange(2, len(ax) -1, 2),  np.arange(len(ax) - 2, len(ax))]:
    ax[i].axis('off') #changed