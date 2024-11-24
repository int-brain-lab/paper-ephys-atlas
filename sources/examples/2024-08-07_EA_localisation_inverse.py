"""
Here we retain the same probes as her testing set for testing and train the model on the remaining probes
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats
import socket

from brainbox.ephys_plots import plot_brain_regions
import iblatlas.atlas
from ibllib.plots import color_cycle

import ephys_atlas.plots
import ephys_atlas.anatomy
import ephys_atlas.data


match socket.gethostname():
    case "little mac":
        # little mac
        LOCAL_DATA_PATH = Path(
            "/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding"
        )
        NEMO_PATH = Path(
            "/Users/olivier/Library/CloudStorage/GoogleDrive-olivier.winter@internationalbrainlab.org/Shared drives/Task Force - Electrophysiology Atlas/Decoding/NEMO"
        )
    case "parede":
        LOCAL_DATA_PATH = Path("/mnt/s0/ephys-atlas-decoding")
        NEMO_PATH = Path(
            "/home/olivier/Insync/olivier.winter@internationalbrainlab.org/Google Drive - Shared drives/Task Force - Electrophysiology Atlas/Decoding/NEMO"
        )
    case "ferret":
        # work off linux
        LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
        NEMO_PATH = Path(
            "/datadisk/team_drives/Task Force - Electrophysiology Atlas/Decoding/NEMO"
        )

model_tag = "2024_W04_Cosmos_03f2362c"  # 0.54 accuracy, void fluid
# '2024_W04_Cosmos_a16bb436'  # 0.53 accuracy, baseline
# '2024_W04_Cosmos_bea38057'  # 0.61 accuracy, no void/root
# '2024_W04_Cosmos_a8b06f2c'  # 0.57 accuracy, void fluid and root inflated


ba = ephys_atlas.anatomy.EncodingAtlas()
br = ba.regions
npz = np.load(NEMO_PATH.joinpath("soft_boundaries.npz"))
brain_template = {
    "soft_boundaries": npz["soft_boundaries"],
    "cosmos_aids": npz["cosmos_aids"],
}

file_results = NEMO_PATH.joinpath(f"voltage_features_with_predictions_{model_tag}.pqt")
df = pd.read_parquet(file_results)
COSMOS_ID = np.sort(df["Cosmos_id"].unique())
_, _, _, df_probes = ephys_atlas.data.load_voltage_features(
    LOCAL_DATA_PATH.joinpath("features", "2024_W04")
)

# %%
pids = ephys_atlas.data.NEMO_TEST_PIDS
pid = pids[57]  # ppt from the 12-08-2024 was made with 67, 6, 123
# 132 is not predictable

df_pid = df.loc[pid]

df_depths = df_pid.groupby("axial_um").mean(numeric_only=True)

columns_labels = ["Cosmos_id", "Cosmos_prediction", "Allen_id"]
daggs = {
    k: pd.NamedAgg(column=k, aggfunc=lambda x: x.mode().iloc[0]) for k in columns_labels
}
df_aids = df_pid.groupby("axial_um").agg(**daggs)
for col in columns_labels:
    df_depths[col] = df_aids[col].values


# %%
fig, axs = plt.subplots(
    1, 6, figsize=(16, 8), gridspec_kw={"width_ratios": [7, 1, 1, 1, 2, 7]}
)
fig.suptitle(f"Probe {pid}")
features_list = ephys_atlas.encoding.voltage_features_set()
sns.heatmap(
    scipy.stats.zscore(df_depths.loc[:, features_list]),
    ax=axs[0],
    vmin=-2,
    vmax=2,
    cmap="Spectral",
    cbar=True,
)
axs[0].set(title="z-scored features")
axs[0].invert_yaxis()
# axs[0].tick_params(axis='x', rotation=45)

plot_brain_regions(
    df_depths["Allen_id"].values,
    channel_depths=df_depths.index.values,
    brain_regions=br,
    display=True,
    ax=axs[1],
    title="Allen",
)
plot_brain_regions(
    df_depths["Cosmos_id"].values,
    channel_depths=df_depths.index.values,
    brain_regions=br,
    display=True,
    ax=axs[2],
    title="Real",
)
plot_brain_regions(
    df_depths["Cosmos_prediction"].values,
    channel_depths=df_depths.index.values,
    brain_regions=br,
    display=True,
    ax=axs[3],
    title="Pred",
)

# ba.regions.to_df().set_index('id').loc[cosmos_aids].reset_index()
pr = brain_template["soft_boundaries"][
    ba._lookup(df_depths.loc[:, ["x", "y", "z"]].values, mode="clip"), :
]
ephys_atlas.plots.plot_cumulative_probas(
    probas=pr,  # (ndepths x nregions) array of probabilities for each region that sum to 1 for each depth
    depths=df_depths.index.values,
    aids=npz["cosmos_aids"],
    regions=ba.regions,
    ax=axs[4],
    legend=False,
)
axs[4].set(title="Soft assignment")

# plot the prediction from the classifier
ephys_atlas.plots.plot_cumulative_probas(
    probas=df_depths.loc[
        :, map(str, list(COSMOS_ID))
    ].values,  # (ndepths x nregions) array of probabilities for each region that sum to 1 for each depth
    depths=df_depths.index.values,
    aids=COSMOS_ID,
    regions=br,
    ax=axs[5],
    legend=False,
)
axs[5].set(title="Classifier prediction")
fig.tight_layout()

# %% The forward problem: given a set of locations get the regions probabilities
import scipy.optimize

y_pred = df_depths.loc[:, map(str, list(brain_template["cosmos_aids"]))].values


# x,y,z : tip electrode coordinates in CCF space
# ML (x, Right: positive), AP (y, A: positive), DV (z, D: positive) stereotaxic axis and zeroed at Bregma, (m)
# phi: azimuth from right (x+) in degrees  [0 - 360]
# theta: polar angle from vertical (z+) in degrees [0 - 180]
# depths: dephts along the shank of the probe (m), starting from 0 at the tip and positive going up the shank
def forward(x, y, z, theta, phi, depths):
    xyz = np.c_[iblatlas.atlas.sph2cart(depths, theta, phi)] + np.array((x, y, z))
    # labels = ba.get_labels(xyz, mapping='Cosmos', mode='clip')
    return xyz


def loss(X):
    # x, y, z, theta, phi, depths = (x0, y0, z0, theta0, phi0, df_depths.index.values / 1e6)
    # y_pred = df_depths['Cosmos_prediction'].values
    x, y, z, theta, phi = X
    xyz = forward(x, y, z, theta, phi, df_depths.index.values / 1e6)
    soft_probas = brain_template["soft_boundaries"][ba._lookup(xyz, mode="clip"), :]
    log_likelihood = np.sum(np.log2(np.sum(y_pred * soft_probas, axis=1)))
    # log_likelihood = np.sum(np.log2(np.sum(np.minimum(y_pred, soft_probas), axis=1)))
    return -log_likelihood


# this is the best possible answer for the set of coordinates
ins = iblatlas.atlas.Insertion.from_track(
    df_pid.loc[:, ["x", "y", "z"]].values, brain_atlas=ba
)
x0, y0, z0, phi0, theta0, d0 = (*ins.tip, ins.phi, ins.theta, ins.depth)
xyz_svd = forward(x0, y0, z0, theta0, phi0, depths=df_depths.index.values / 1e6)
ba.regions.id[
    ba.regions.mappings["Cosmos"][ba.label.flatten()[ba._lookup(xyz_svd, mode="clip")]]
]


# this is the planned trajectory, what we will use in practice
ins_planned = iblatlas.atlas.Insertion.from_dict(
    dict(
        x=df_probes.loc[pid].x_target,
        y=df_probes.loc[pid].y_target,
        z=df_probes.loc[pid].z_target,
        phi=df_probes.loc[pid].phi_target,
        theta=df_probes.loc[pid].theta_target,
        depth=df_probes.loc[pid].depth_target,
    ),
    brain_atlas=ba,
)
x0, y0, z0, phi0, theta0 = (*ins_planned.tip, ins_planned.phi, ins_planned.theta)
xyz_planned = forward(x0, y0, z0, theta0, phi0, depths=df_depths.index.values / 1e6)

X0 = (*ins_planned.tip, ins_planned.phi, ins_planned.theta)
x, y, z, theta, phi = X0
opt = scipy.optimize.minimize(loss, x0=X0, method="CG")

opt = scipy.optimize.differential_evolution(
    loss,
    bounds=(
        np.array([-1, 1]) * 1e-3 + X0[0],
        np.array([-1, 1]) * 1e-3 + X0[1],
        np.array([-1, 1]) * 1e-3 + X0[2],
        np.array([-20, 20]) + X0[3],
        np.array([-10, 10]) + X0[4],
    ),
)

xyz_opt = forward(*opt.x, depths=-df_depths.index.values / 1e6)


# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle(f"Probe {pid}")
ba.plot_cslice(
    ap_coordinate=np.median(df_pid["y"]), volume="annotation", ax=axs[0], alpha=0.7
)
ba.plot_sslice(
    ml_coordinate=np.median(df_pid["x"]), volume="annotation", ax=axs[1], alpha=0.7
)
plot_args = {"linewidth": 2}

axs[0].plot(opt.x[0] * 1e6, opt.x[2] * 1e6, "r*")
axs[1].plot(opt.x[1] * 1e6, opt.x[2] * 1e6, "r*")

axs[0].plot(xyz_opt[:, 0] * 1e6, xyz_opt[:, 2] * 1e6, label="optimized", **plot_args)
axs[1].plot(xyz_opt[:, 1] * 1e6, xyz_opt[:, 2] * 1e6, label="optimized", **plot_args)
axs[0].plot(
    df_pid["x"].values * 1e6,
    df_pid["z"].values * 1e6,
    ".r",
    label="aligned channels",
    **plot_args,
)
axs[1].plot(
    df_pid["y"].values * 1e6,
    df_pid["z"].values * 1e6,
    ".r",
    label="aligned channels",
    **plot_args,
)

axs[0].plot(
    ins_planned.xyz[:, 0] * 1e6,
    ins_planned.xyz[:, 2] * 1e6,
    label="planned trajectory",
    **plot_args,
)
axs[1].plot(
    ins_planned.xyz[:, 1] * 1e6,
    ins_planned.xyz[:, 2] * 1e6,
    label="planned trajectory",
    **plot_args,
)
axs[0].plot(
    xyz_svd[:, 0] * 1e6, xyz_svd[:, 2] * 1e6, label="forward trajectory", **plot_args
)
axs[1].plot(
    xyz_svd[:, 1] * 1e6, xyz_svd[:, 2] * 1e6, label="forward trajectory", **plot_args
)

axs[0].legend()
axs[1].legend()
fig.tight_layout()


# %% Visualize part of the cost function
X0 = (*ins.tip, ins.phi, ins.theta)  # regression of histology channel locations

plt.figure()

x = np.linspace(-1, 1, 250) * 1e-3
y = np.zeros_like(x)
X = np.array(X0)
for i, xi in enumerate(x):
    X[0] = X0[0] + xi
    y[i] = loss(X)
plt.plot(x * 1e6, y)


X = np.array(X0)
for i, xi in enumerate(x):
    X[1] = X0[1] + xi
    y[i] = loss(X)
plt.plot(x * 1e6, y)


X = np.array(X0)
for i, xi in enumerate(x):
    X[2] = X0[2] + xi
    y[i] = loss(X)
plt.plot(x * 1e6, y)


# %%
from mayavi import mlab
from iblapps.atlaselectrophysiology import rendering

fig = rendering.figure(grid=False)
mlapdv = ba.xyz2ccf(df_pid.loc[:, ["x", "y", "z"]].values)
mlab.plot3d(
    mlapdv[:, 1],
    mlapdv[:, 2],
    mlapdv[:, 0],
    line_width=1,
    tube_radius=20,
    color=color_cycle(0),
)
mlab.text3d(
    mlapdv[-1, 1],
    mlapdv[-1, 2],
    mlapdv[-1, 0] - 500,
    pid[:8],
    line_width=4,
    color=color_cycle(0),
    figure=fig,
    scale=150,
)
mlapdv = ba.xyz2ccf(xyz, mode="clip")
mlab.plot3d(
    mlapdv[:, 1],
    mlapdv[:, 2],
    mlapdv[:, 0],
    line_width=1,
    tube_radius=20,
    color=color_cycle(1),
)


# %%
"""

new thread since the one above got a bit too long :slightly_smiling_face:
here’s a geometrically aware model that i think should improve on the hmm, and would take better advantages of landmarks like the large fiber tracts, ventricles, etc.
we have a template brain, plus “edge” variables (one for each pair of brain regions that touch each other), and the 5-dimensional probe coordinates (3d for the tip plus 2d for the angles).
the “edge” variables are there to model brain variability - ie the boundary between ctx and hc is typically not exactly where the template brain says it should be.
most of these edge variables can be ignored since they are far from the probe.
we want to optimize over the probe coordinates and the relevant subset of edge coordinates.  the objective function is the summed loglikelihood of each per-channel feature observation - we can pre-compute these likelihoods just as 
@Han Yu
 did when combining her decoder with 
@Olivier
 ’s.  we can add a regularization penalty to this to prevent the edge variables from getting too far from their baseline (ie how far the observed brain can be from the template).
so now we would just do gradient descent on this objective function, initialized from the targeted probe coordinates (and the default values of the edge variables).  this is pretty similar to what humans would do in the registration step if they don’t have access to the histology image - start with the targeted coordinates and then adjust these coordinates and the region boundaries a bit to match the observed data.
@Olivier
 
@Han Yu
 wdyt?
 
 
writing down the objective function following the discussion on M.
first compute log p(region | features) for each decoder.  this is an RxC matrix; call it L.
there’s also a std dev s that defines the optimal precision of the localization.
now the objective function is a function of the probe location and the edge locations.  for each channel c,
 we can compute a soft assignment of the brain region (computed by smoothing the edge with a gaussian of width s).
so we get an Rx1 soft assignment vector a_c for each c. now we just compute the sum:
sum_rc a_c(r) L(r,c)
note that the soft-assign vectors are continuous+differentiable as a function of the probe location and edge location params, so the objective function is differentiable too

9:18
if there are too many local optima then we could start with an enlarged s (ie oversmooth a bit to reduce the number of local optima) and then reduce s to the true value during the optimization
"""
