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

from iblutil.numerical import ismember
import matplotlib.pyplot as plt
from iblatlas.atlas import get_bc, BrainAtlas, aws, AllenAtlas
from iblatlas.regions import BrainRegions

def _download_depth_files(file_name):
    """
    Download and return path to relevant file
    :param file_name:
    :return:
    """
    file_path = BrainAtlas._get_cache_dir().joinpath('depths', file_name)
    if not file_path.exists():
        file_path.parent.mkdir(exist_ok=True, parents=True)
        aws.s3_download_file(f'atlas/depths/{file_path.name}', file_path)

    return file_path
def xyz_to_depth(xyz, per=True, res_um=25):
    """
    For a given xyz coordinates return the depth from the surface of the cortex. The depth is returned
    as a percentage if per=True and in um if per=False. Note the lookup will only work for xyz cooordinates
    that are in the Isocortex of the Allen volume. If coordinates outside of this region are given then
    the depth is returned as nan.

    Parameters
    ----------
    xyz : numpy.array
        An (n, 3) array of Cartesian coordinates. The order is ML, AP, DV and coordinates should be given in meters
        relative to bregma.

    per : bool
        Whether to do the lookup in percentage from the surface of the cortex or depth in um from the surface of the cortex.

    res_um : float or int
        The resolution of the brain atlas to do the depth lookup

    Returns
    -------
        numpy.array
        The depths from the surface of the cortex for each cartesian coordinate. If the coordinate does not lie within
        the Isocortex, depth value returned is nan
    """

    ind_flat = np.load(_download_depth_files(f'depths_ind_{res_um}.npy'))
    depth_file = f'depths_per_{res_um}.npy' if per else f'depths_um_{res_um}.npy'
    depths = np.load(_download_depth_files(depth_file))
    bc = get_bc(res_um=res_um)

    ixyz = bc.xyz2i(xyz, mode='clip')
    iravel = np.ravel_multi_index((ixyz[:, 1], ixyz[:, 0], ixyz[:, 2]), (bc.ny, bc.nx, bc.nz))
    a, b = ismember(iravel, ind_flat)

    lookup_depths = np.full(iravel.shape, np.nan, dtype=np.float32)
    lookup_depths[a] = depths[b]

    return lookup_depths

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
# df_voltage['cortical_depth'] = iblatlas.atlas.xyz_to_depth(df_voltage.loc[:, ['x', 'y', 'z']].values)
df_voltage['cortical_depth'] = xyz_to_depth(df_voltage.loc[:, ['x', 'y', 'z']].values, per=True)



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
# %%
