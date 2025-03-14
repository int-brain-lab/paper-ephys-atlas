"""3D brain with insertions, using Datoviz.

NOTES:
- currently requires a development version of Datoviz (i.e. custom-compiled)
- requires `insertions.pqt` and `brain.obj` in the directory (not staged in the repo atm)

"""

from iblatlas.atlas import NeedlesAtlas, AllenAtlas
from pprint import pprint
from pathlib import Path

import pandas as pd
import numpy as np
import datoviz as dvz
from datoviz import vec3, vec4, S_, A_


CURDIR = Path(__file__).parent


def df2insertions(df):
    """Transform a dataframe with channel insertions to a `((n_insertions, n_channels, 3)` array
    with insertion trajectories."""
    ba = AllenAtlas(25)
    dfr = df.reset_index().pivot(
        index='channel', columns='pid', values=['x', 'y', 'z'])
    # This is (n_channels, 3, n_insertions)
    pos = dfr.values.reshape(dfr.shape[0], 3, -1)
    n_channels, _, n_insertions = pos.shape
    posr = np.vstack([pos[:, :, i] for i in range(n_insertions)])
    posr = posr.reshape((-1, 3))  # (n_channels * n_insertions, 3)
    # NOTE: note the ccf_order
    ccf = ba.xyz2ccf(posr, mode='clip', ccf_order='apdvml')
    ccf = ccf.reshape((n_insertions, n_channels, 3))
    idx = np.all(np.all(~np.isnan(ccf), axis=1), axis=1)
    ccf = ccf[idx]
    ccf = np.ascontiguousarray(ccf)
    return ccf


def load_brain():
    filepath = (CURDIR / "brain.obj").resolve()
    shape = dvz.shape_obj(S_(filepath))
    return shape


class Axes:
    app = None
    batch = None
    scene = None
    figure = None
    panel = None
    vtype = 'basic'
    ARCBALL = vec3(-2.76, -0.57, -0.25)

    def __init__(self):
        self.app = dvz.app(dvz.APP_FLAGS_WHITE_BACKGROUND)
        self.batch = dvz.app_batch(self.app)
        self.scene = dvz.scene(self.batch)
        self.figure = dvz.figure(self.scene, 800, 600, dvz.CANVAS_FLAGS_IMGUI)
        self.panel = dvz.panel_default(self.figure)
        self.arcball = dvz.panel_arcball(self.panel)
        self.camera = dvz.panel_camera(self.panel, 0)

        # dvz.arcball_gui(
        #     self.arcball, self.app, dvz.figure_id(self.figure), self.panel)

        dvz.camera_initial(
            self.camera, vec3(0, 0, 3), vec3(0, 0, 0), vec3(0, 1, 0))
        dvz.arcball_initial(self.arcball, self.ARCBALL)
        dvz.panel_update(self.panel)

        self._on_timer = dvz.timer(self._on_timer)
        dvz.app_ontimer(self.app, self._on_timer, None)

    def insertions(self, insertions):
        vtype = self.vtype
        batch = self.batch

        n_insertions, n_channels, _ = insertions.shape
        n = n_channels * n_insertions

        if vtype == 'basic':
            visual = dvz.basic(batch, dvz.PRIMITIVE_TOPOLOGY_LINE_STRIP, 0)
        else:
            visual = dvz.path(batch, 0)
        dvz.visual_depth(visual, dvz.DEPTH_TEST_ENABLE)

        if vtype == 'basic':
            dvz.basic_alloc(visual, n)
        else:
            dvz.path_alloc(visual, n)

        pos = np.ascontiguousarray(
            insertions.reshape((-1, 3)), dtype=np.float32)
        if vtype == 'basic':
            dvz.basic_position(visual, 0, n, pos, 0)
        else:
            group_sizes = np.full(n_insertions, n_channels).astype(np.uint32)
            dvz.path_position(visual, n, pos, n_insertions, group_sizes, 0)

        if vtype != 'basic':
            dvz.path_linewidth(visual, 4)
            dvz.path_join(visual, dvz.JOIN_ROUND)

        # Groups.
        if vtype == 'basic':
            group_sizes = np.full(n_insertions, n_channels).astype(np.uint32)
            dvz.visual_groups(visual, n_insertions, group_sizes)
            group = np.repeat(np.arange(n_insertions),
                              n_channels).astype(np.float32)
            dvz.basic_group(visual, 0, n, group, 0)

        color = np.empty((n, 4), dtype=np.uint8)
        t = np.repeat(np.linspace(0, 1, n_insertions),
                      n_channels).astype(np.float32)
        dvz.colormap_array(dvz.CMAP_PLASMA, n, t, 0, 1, color)
        # color[:, 3] = 128
        if vtype == 'basic':
            dvz.basic_color(visual, 0, n, color, 0)
        else:
            dvz.path_color(visual, 0, n, color, 0)

        dvz.panel_visual(self.panel, visual, 0)

    def brain(self, brain):
        shape = brain
        batch = self.batch
        panel = self.panel

        nv = shape.vertex_count
        ni = shape.index_count

        flags = dvz.MESH_FLAGS_LIGHTING
        mesh = dvz.mesh_shape(batch, shape, flags)

        colors = np.full((nv, 4), 255, dtype=np.uint8)
        colors[:, 3] = 128
        dvz.mesh_color(mesh, 0, nv, colors, 0)

        dvz.visual_depth(mesh, dvz.DEPTH_TEST_DISABLE)
        dvz.visual_cull(mesh, dvz.CULL_MODE_BACK)
        dvz.visual_blend(mesh, dvz.BLEND_OIT)
        light_params = vec4(.8, .2, .2, 8)
        dvz.mesh_light_params(mesh, 0, light_params)

        dvz.panel_visual(panel, mesh, 0)

    def _on_timer(self, app, window_id, ev):
        arcball = self.ARCBALL
        arcball[1] += .01
        dvz.arcball_set(self.arcball, arcball)
        dvz.panel_update(self.panel)

    def run(self):
        # NOTE: disable the automatic rotation for now
        # dvz.app_timer(self.app, 0, 1. / 60., 0)
        dvz.scene_run(self.scene, self.app, 0)
        dvz.scene_destroy(self.scene)
        dvz.app_destroy(self.app)


if __name__ == '__main__':

    brain = load_brain()

    df = pd.read_parquet('channels.pqt')
    insertions = df2insertions(df)

    pos_brain = dvz.pointer_array(
        brain.pos, brain.vertex_count, 3, dtype=np.float32)

    center = pos_brain.mean(axis=0)
    pos_brain -= center
    insertions -= center

    pos = np.vstack((pos_brain, insertions.reshape((-1, 3))))
    m = pos.min(axis=0)
    M = pos.max(axis=0)
    k = np.maximum(np.abs(m), np.abs(M)).max()
    pos_brain /= k
    insertions /= k

    ax = Axes()
    ax.brain(brain)
    ax.insertions(insertions)
    ax.run()


##############################################

"""

# fn = CURDIR / 'brain_insertions.npz'
# if not fn.exists():
# print("Computing brain and insertions")
    # np.savez(fn, dict(pos_brain=pos_brain, insertions=insertions))
# else:
#     data = np.load(fn)
#     brain = data['brain']
#     insertions = data['insertions']


# Colors.
# color = np.full((n, 4), 255).astype(np.uint8)
# color[:, 3] = 255
# dvz.basic_color(visual, 0, n, color, 0)
# Brain mesh


# pos = xzy_channels_table * .001  # (n_channels, n_insertions, 3)
# pos = pos.astype(np.float32)
# xzy_channels_table = df[['x', 'y', 'z']].to_numpy()
# xzy_channels_table = xzy_channels_table.reshape(n_insertions, 3, -1)
# df_channels = pd.read_parquet('channels.pqt')


# df_channels = pd.read_parquet('channels.pqt')
# reshaped_df = df_channels.reset_index().pivot(index='channel', columns='pid', values=['x', 'y', 'z'])
# pos = reshaped_df.values.reshape(reshaped_df.shape[0], 3, -1)
# print(pos.shape)
# np.save('insertions.npy', pos.astype(np.float32))

# pos = np.load('insertions.npy')
# n_channels, _, n_insertions = pos.shape

# pos2 = np.vstack([pos[:, :, i] for i in range(n_insertions)])
# pos2 = pos2.reshape((-1, 3))

# xzy_channels_table = ba.xyz2ccf(pos2, mode='clip')
# xzy_channels_table = xzy_channels_table.reshape((n_channels, n_insertions, 3))
# print(xzy_channels_table.shape)
# plot(xzy_channels_table[:, 0, 0], xzy_channels_table[:, 0, 1])
# exit()

# df_channels = pd.read_parquet('channels.pqt')
# xzy_channels_table = df_channels[['x', 'y', 'z']].to_numpy()
# xzy_channels_table = xzy_channels_table.reshape(n_insertions, 3, -1)
# np.save('insertions2.npy', xzy_channels_table.astype(np.float32))

# pos = np.load('insertions2.npy')
# print(pos.shape, pos.dtype, pos)
# exit()

# pos = xzy_channels_table * .001  # (n_channels, n_insertions, 3)
# pos = pos.astype(np.float32)

# insertions = np.all(np.all(~np.isnan(pos), axis=0), axis=0)
# pos = pos[::4, insertions, :]
# n_channels, _, n_insertions = pos.shape

# Data normalization.
# pos[np.isnan(pos)] = 0
# me = pos.mean(axis=0).mean(axis=-1)
# pos -= me.reshape((1, 3, 1))
# M = max(abs(pos.min()), abs(pos.max()))
# pos /= M


######


# from pathlib import Path
# from one.api import ONE
# import ephys_atlas.data
# import seaborn as sns
# from mayavi import mlab
# from atlaselectrophysiology import rendering
# from ibllib.atlas import NeedlesAtlas, AllenAtlas
# from brainwidemap import bwm_query


# sns.set_theme(context='talk')

# needles = NeedlesAtlas()
# ba = AllenAtlas(25)
# ba.compute_surface()

# LOCAL_DATA_PATH = Path("/Users/gaellechapuis/Documents/Work/Secondpass/Report")
# save_fig_path = LOCAL_DATA_PATH.joinpath('pics')

# # data loading section
# one = ONE()
# one.alyx.clear_rest_cache()  # reset cache
# path_features = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/')
# label = 'latest'
# # df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(path_features,
# #                                                                                    label=label,
# #                                                                                    one=one)

# df_channels = pd.read_parquet('channels.pqt')
# pids = df_channels.index.get_level_values(0).unique()

# ##
# # Use external database to get BW PIDs
# one2 = ONE(base_url='https://openalyx.internationalbrainlab.org',
#           password='international', silent=True)
# df_bw = bwm_query(one2)
# pids_bw = df_bw['pid'].to_list()
# ##
# fig = rendering.figure()

# for pid in pids:
#     # Take channel coordinates value (mm) from table, convert into Allen coordinate system
#     pid_df = df_channels.loc[pid]
#     xzy_channels_table = pid_df[['x', 'y', 'z']].to_numpy()
#     try:
#         xzy_channels_table = ba.xyz2ccf(xzy_channels_table)
#         # Plot in 3D
#         if pid in pids_bw:
#             color = (0.0, 0.2, 0.6)  # blue
#         else:
#             color = (0.6, 0.2, 0.2)  # red

#         mlab.points3d(xzy_channels_table[:, 1], xzy_channels_table[:, 2], xzy_channels_table[:, 0],
#                       color=color,
#                       mode="point",
#                       scale_mode="none",
#                       scale_factor=1.0
#                       )
#     except:
#         print(f'ERROR - {pid}')

# mlab.pitch(50)
# mlab.savefig(filename='test.png')
"""
