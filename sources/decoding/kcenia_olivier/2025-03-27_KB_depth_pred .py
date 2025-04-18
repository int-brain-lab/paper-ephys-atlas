#%%
"""
    27March2025
    OW, KB
    Predicting a continuous variable - cortical depth
    Merging OW [get data and example model] and MF [atlas acronym / id and depth function] codes
"""

# %%
import matplotlib.pyplot as plt
from pathlib import Path
from one.api import ONE
import ephys_atlas.data 
from pathlib import Path
import pandas as pd
import numpy as np

import sklearn.metrics
from xgboost import XGBClassifier  # pip install xgboost  # https://xgboost.readthedocs.io/en/stable/prediction.html

from iblutil.numerical import ismember
import ephys_atlas.encoding
import ephys_atlas.decoding
import ephys_atlas.anatomy
import ephys_atlas.data
import ephys_atlas.features
# from iblatlas.atlas import xyz_to_depth


import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor


import brainbox.ephys_plots
import iblatlas.atlas
import iblatlas.plots
import matplotlib

import iblatlas.atlas
from iblatlas.gui import atlasview

from iblatlas.atlas import BrainRegions

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


# %%
LOCAL_DATA_PATH = Path.home().joinpath("Downloads")
LABEL = "2024_W50"  # or put "latest" # or '2024_W50'
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)

# %%
# Here we prepare the dataframe by selecting only the records in isocortex, and computing the cortical depth

df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)

df_voltage['cortical_depths']  = xyz_to_depth(df_voltage[['x', 'y', 'z']].to_numpy(), per=True) #change here for percentage per=True
df_voltage = df_voltage.dropna().copy()

print(df_voltage.dropna().shape)

# %%
x_list = ephys_atlas.features.voltage_features_set(features_list= ['raw_ap', 'raw_lf', 'raw_lf_csd', 'waveforms'])
X = df_voltage.loc[:, x_list].values
scaler = StandardScaler()
scaler.fit(X)
stratify = df_voltage.index.get_level_values("pid").to_numpy()
Y = df_voltage['cortical_depths'].to_numpy()

train_idx, test_idx = ephys_atlas.encoding.train_test_split_indices(df_voltage)
X_test, y_test = (scaler.transform(X[test_idx, :]), Y[test_idx])
X_train, y_train = (scaler.transform(X[train_idx, :]), Y[train_idx])

# %%
# 3. Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

# %%
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%% 
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%%
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%% 
from sklearn.svm import SVR

model = SVR(kernel='rbf')  # or 'linear', 'poly'
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')

#%% 
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# 3. Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "BayesianRidge": BayesianRidge(),
    "HuberRegressor": HuberRegressor()
}

# 4. Fit & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}: R² = {r2:.4f}, MSE = {mse:.4f}")

#%% 
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)
print(f"MLPRegressor: R² = {r2_score(y_test, y_pred):.4f}, MSE = {mean_squared_error(y_test, y_pred):.4f}")

#%%
"""
PLOT
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel('Actual Depth')
plt.ylabel('Predicted Depth')
plt.title('Linear Regression: Predicted vs Actual Depth')
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
scatter = plt.scatter(y_test, y_pred, c=y_test, cmap='viridis', alpha=0.02)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Prediction')
plt.gca().set(ylim=[-50, 1000])
plt.xlabel('Actual Depth')
plt.ylabel('Predicted Depth')
plt.title('Predicted vs Actual Depth (Color = True Depth)')
plt.colorbar(scatter, label='Actual Depth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
CODE TO PLOT BRAIN REGIONS AND A FEATURE SIDE BY SIDE - WE WANT THE DEPTH IN THE FEATURE PART
""" 
import sys
sys.path.append('/mnt/h0/kb/code_kcenia/ibllib')

# Now you can import anything from ibllib, like:
import brainbox
from brainbox import ephys_plots
from iblatlas.regions import BrainRegions


pids = df_voltage.index.get_level_values(0).unique().tolist()

for pid in pids:
    print(pid)
    pid_df = df_voltage.loc[pid]

first_pid = pids[0]
df_voltage_test01 = df_voltage.loc[first_pid]
channel_ids = df_voltage_test01.atlas_id


br = BrainRegions()
region_info = br.get(channel_ids)
boundaries = np.where(np.diff(region_info.id) != 0)[0]
boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]

regions = np.c_[boundaries[0:-1], boundaries[1:]]





for i, pid in enumerate(pids[:5]):
    # TODO label with the pids
    n_pids = 5
    fig, ax = plt.subplots(n_pids, 3, figsize=(12, 4 * n_pids))
    df_pid = df_voltage.loc[pid]
    depth = df_pid.groupby('axial_um').agg(atlas_id=pd.NamedAgg(column='atlas_id', aggfunc='first')).reset_index()
    brainbox.ephys_plots.plot_brain_regions(depth['atlas_id'].values, channel_depths=depth['axial_um'].values,
                                            brain_regions=regions, display=True, ax=ax[i * 3])
    ax[-2].plot(df_pid['x'].values * 1e6, df_pid['y'].values * 1e6, '.', color='r', markersize=2)
    ax[-2].plot(df_pid['x'].values[-1] * 1e6, df_pid['y'].values[-1] * 1e6, '*', color='k', markersize=2)
    ax[i * 3 + 2].axis('off')


    feature = df_pid["depths"].values
    if feature_name in transform:
        feature = transform["depths"](feature)
    data_bank, x_bank, y_bank = brainbox.plot_base.arrange_channels2banks(
        data=feature,
        chn_coords=df_pid[['lateral_um', 'axial_um']].values,
        pad=True,
    )
    data = brainbox.plot_base.ProbePlot(data_bank, x=x_bank, y=y_bank, cmap=colormap)
    data.clim = clim
    brainbox.ephys_plots.plot_probe(data.convert2dict(), ax=ax[i * 3 + 1], show_cbar=False)
    ax[i * 3 + 1].axis('off')
# %%
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import ephys_atlas.data
import brainbox.ephys_plots
import iblatlas.atlas
import iblatlas.plots

def figure_style():
    sns.set(style="ticks", context="paper",
            rc={"font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.title_fontsize": 7,
                "lines.linewidth": 1,
                "lines.markersize": 4,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "savefig.transparent": False,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

MM_TO_INCH = 1 / 25.4

figure_style()
FIG_SIZE = (13, 6)

def adjust_figure(fig):
    # Remove 7.5 mm of whitespace around figure in all directions
    adjust = 7.5
    # Depending on the location of axis labels leave a bit more space
    extra =  5
    width, height = fig.get_size_inches() / MM_TO_INCH
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + extra)/height,
                        left=(adjust + extra)/width, right=1-adjust/width)


regions = iblatlas.atlas.BrainRegions()
ba = iblatlas.atlas.AllenAtlas()
pids = ephys_atlas.data.BENCHMARK_PIDS
n_pids = len(pids)

# Plot the brain regioons for the set of pids
fig, ax = plt.subplots(1, 15, figsize=FIG_SIZE, gridspec_kw={'width_ratios': np.r_[np.ones(n_pids), 1.5, 8]})
ba.plot_top(ax=ax[-1])


# # Select only the first 5 pids
# pids = df_voltage.index.get_level_values(0).unique().tolist()[:5]
# n_pids = len(pids)

for i, pid in enumerate(pids):
    df_pid = df_voltage.loc[pid]
    depth = df_pid.groupby('axial_um').agg(atlas_id=pd.NamedAgg(column='atlas_id', aggfunc='first')).reset_index()
    brainbox.ephys_plots.plot_brain_regions(depth['atlas_id'].values, channel_depths=depth['axial_um'].values, brain_regions=regions, display=True, ax=ax[i])
    ax[-1].plot(df_pid['x'].values * 1e6, df_pid['y'].values * 1e6, '.', color='r', markersize=2)
    ax[-1].plot(df_pid['x'].values[-1] * 1e6, df_pid['y'].values[-1] * 1e6, '*', color='k', markersize=2)

plt.show()
for i in range(13, 15):
    ax[i].axis('off')
adjust_figure(fig)


# %% Plot the features for the selected pid

fig, ax = plt.subplots(1, 15, figsize=FIG_SIZE, gridspec_kw={'width_ratios': np.r_[np.ones(n_pids), 1.5, 8]})

feature_name = 'trough_val'
colormap = 'PuOr_r'
feature = df_voltage[feature_name].values
if feature_name in transform:
    feature = transform[feature_name](feature)
clim = np.array([np.nanquantile(feature, 0.1), np.nanquantile(feature, 0.9)])
hlim = np.array([np.nanquantile(feature, 0.01), np.nanquantile(feature, 0.99)])

# np.diff(quants)
c, x = np.histogram(feature, bins=np.linspace(hlim[0], hlim[1], 64))
bars = ax[-1].bar(x[:-1], c / np.sum(c), width=np.diff(x)[0])
cmap = plt.get_cmap(colormap)  # You can choose any colormap you prefer
norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
for bar, bin_center in zip(bars, x[:-1]):
    bar.set_color(cmap(norm(bin_center)))

for i, pid in enumerate(pids):
    df_pid = df_voltage.loc[pid]
    feature = df_pid[feature_name].values
    if feature_name in transform:
        feature = transform[feature_name](feature)
    data_bank, x_bank, y_bank = brainbox.plot_base.arrange_channels2banks(
        data=feature,
        chn_coords=df_pid[['lateral_um', 'axial_um']].values,
        pad=True,
    )
    data = brainbox.plot_base.ProbePlot(data_bank, x=x_bank, y=y_bank, cmap=colormap)
    data.clim = clim
    brainbox.ephys_plots.plot_probe(data.convert2dict(), ax=ax[i], show_cbar=False)
    if i > 0:
        # turn off axis labels
        # ax[i].set_yticklabels([])
        ax[i].axis('off')

ax[0].set(ylabel='Depth (um)')
ax[-2].axis('off')
fig.suptitle(f'{feature_name}')
plt.show()
adjust_figure(fig)

# %% double trouble
fig, ax = plt.subplots(1, n_pids * 3  + 2, figsize=FIG_SIZE,
                       gridspec_kw={'width_ratios': np.r_[np.ones(n_pids * 3), 1.5, 8]})
ba.plot_top(ax=ax[-1])

#%% 
# Set up figure and axes once
fig, ax = plt.subplots(n_pids, 3, figsize=(1, 4 * n_pids))
ax = ax.flatten()

br = BrainRegions()

# Use first PID to define regions (you already did this)
channel_ids = df_voltage.loc[pids[0]].atlas_id
region_info = br.get(channel_ids)
boundaries = np.where(np.diff(region_info.id) != 0)[0]
boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
regions = np.c_[boundaries[0:-1], boundaries[1:]]

# Now loop through the 5 pids
for i, pid in enumerate(pids):
    df_pid = df_voltage.loc[pid]
    
    # Plot brain regions
    depth = df_pid.groupby('axial_um').agg(
        atlas_id=pd.NamedAgg(column='atlas_id', aggfunc='first')
    ).reset_index()
    
    brainbox.ephys_plots.plot_brain_regions(
        depth['atlas_id'].values,
        channel_depths=depth['axial_um'].values,
        brain_regions=br,
        display=True,
        ax=ax[i * 3]
    )
    ax[i * 3].set_title(f"Regions - {pid[:8]}")

    # Plot 2D coordinates
    ax[i * 3 + 2].plot(df_pid['x'].values * 1e6, df_pid['y'].values * 1e6, '.', color='r', markersize=2)
    ax[i * 3 + 2].plot(df_pid['x'].values[-1] * 1e6, df_pid['y'].values[-1] * 1e6, '*', color='k', markersize=2)
    ax[i * 3 + 2].axis('off')

    # Plot feature with depth
    feature = df_pid["cortical_depths"].values

    data_bank, x_bank, y_bank = brainbox.plot_base.arrange_channels2banks(
        data=feature,
        chn_coords=df_pid[['lateral_um', 'axial_um']].values,
        pad=True,
    )
    data = brainbox.plot_base.ProbePlot(data_bank, x=x_bank, y=y_bank, cmap='coolwarm')
    # data.clim = clim
    # brainbox.ephys_plots.plot_probe(data.convert2dict(), ax=ax[i * 3 + 1], show_cbar=False)
    ax[i * 3 + 1].axis('off')

# Optional: adjust layout and show/save
plt.tight_layout()
plt.show()

# %%
##############################################
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def figure_style():
    sns.set(style="ticks", context="paper",
            rc={"font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.title_fontsize": 7,
                "lines.linewidth": 1,
                "lines.markersize": 4,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "savefig.transparent": False,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


import ephys_atlas.data
import brainbox.ephys_plots
import iblatlas.atlas
import iblatlas.plots
import matplotlib
from iblatlas.atlas import xyz_to_depth
from one.api import ONE



MM_TO_INCH = 1 / 25.4

transform = {
    'polarity': lambda x: np.log10(x + 1 + 1e-6),
    'rms_lf': lambda x: 20 * np.log10(x),
    'rms_ap': lambda x: 20 * np.log10(x),
    'rms_lf_csd': lambda x: 20 * np.log10(x),
    'spike_count': lambda x: np.log10(x),
}

figure_style()
FIG_SIZE = (13, 6)




LOCAL_DATA_PATH = Path.home().joinpath("Downloads")
LABEL = "2024_W50"  # or put "latest" # or '2024_W50'
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)

df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)

df_voltage['cortical_depths']  = xyz_to_depth(df_voltage[['x', 'y', 'z']].to_numpy())



regions = iblatlas.atlas.BrainRegions()
ba = iblatlas.atlas.AllenAtlas()
pids = df_voltage.index.get_level_values(0).unique().tolist()[:3]
n_pids = len(pids)


# %% Plot the brain regioons for the set of pids
widths = np.r_[np.ones(n_pids), 1.5, 8]
fig, ax = plt.subplots(1, len(widths), figsize=FIG_SIZE, gridspec_kw={'width_ratios': widths})
ba.plot_top(ax=ax[-1])

for i, pid in enumerate(pids):
    df_pid = df_voltage.loc[pid]
    depth = df_pid.groupby('axial_um').agg(atlas_id=pd.NamedAgg(column='atlas_id_target', aggfunc='first')).reset_index()
    brainbox.ephys_plots.plot_brain_regions(depth['atlas_id'].values, channel_depths=depth['axial_um'].values, brain_regions=regions, display=True, ax=ax[i])
    ax[-1].plot(df_pid['x'].values * 1e6, df_pid['y'].values * 1e6, '.', color='r', markersize=2)
    ax[-1].plot(df_pid['x'].values[-1] * 1e6, df_pid['y'].values[-1] * 1e6, '*', color='k', markersize=2)

plt.show()
for i in range(len(ax) - 2, len(ax)):
    ax[i].axis('off') #changed

# %% Plot the features for the selected pid
widths = np.r_[np.ones(n_pids), 1.5, 8]
fig, ax = plt.subplots(1, len(widths), figsize=FIG_SIZE, gridspec_kw={'width_ratios': widths})

feature_name = 'cortical_depths'
colormap = 'PuOr_r'
feature = df_voltage[feature_name].values
if feature_name in transform:
    feature = transform[feature_name](feature)
clim = np.array([np.nanquantile(feature, 0.1), np.nanquantile(feature, 0.9)])
hlim = np.array([np.nanquantile(feature, 0.01), np.nanquantile(feature, 0.99)])

# np.diff(quants)
c, x = np.histogram(feature, bins=np.linspace(hlim[0], hlim[1], 64))
bars = ax[-1].bar(x[:-1], c / np.sum(c), width=np.diff(x)[0])
cmap = plt.get_cmap(colormap)  # You can choose any colormap you prefer
norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
for bar, bin_center in zip(bars, x[:-1]):
    bar.set_color(cmap(norm(bin_center)))

for i, pid in enumerate(pids):
    df_pid = df_voltage.loc[pid]
    feature = df_pid[feature_name].values
    if feature_name in transform:
        feature = transform[feature_name](feature)
    data_bank, x_bank, y_bank = brainbox.plot_base.arrange_channels2banks(
        data=feature,
        chn_coords=df_pid[['lateral_um', 'axial_um']].values,
        pad=True,
    )
    data = brainbox.plot_base.ProbePlot(data_bank, x=x_bank, y=y_bank, cmap=colormap)
    data.clim = clim
    brainbox.ephys_plots.plot_probe(data.convert2dict(), ax=ax[i], show_cbar=False)
    if i > 0:
        # turn off axis labels
        # ax[i].set_yticklabels([])
        ax[i].axis('off')

ax[0].set(ylabel='Depth (um)')
ax[-2].axis('off')
fig.suptitle(f'{feature_name}')
plt.show()