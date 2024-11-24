from pathlib import Path

import pandas as pd
import numpy as np

import ephys_atlas.data
import ephys_atlas.plots
from iblatlas.regions import BrainRegions
from one.api import ONE


one = ONE(mode="remote")
regions = BrainRegions()
local_path = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
# local_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Data/")
# local_path = Path("/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
label = "2023_W41"
# df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(local_path, label=label, one=one, verify=True)
df_voltage, df_clusters, df_channels, df_probes = (
    ephys_atlas.data.load_voltage_features(local_path.joinpath(label))
)

x_list = [
    "rms_ap",
    "alpha_mean",
    "alpha_std",
    "spike_count",
    "rms_lf",
    "psd_delta",
    "psd_theta",
    "psd_alpha",
    "psd_beta",
    "psd_gamma",
]
x_list += [
    "peak_time_secs",
    "peak_val",
    "trough_time_secs",
    "trough_val",
    "tip_time_secs",
    "tip_val",
]
x_list += [
    "polarity",
    "depolarisation_slope",
    "repolarisation_slope",
    "recovery_time_secs",
    "recovery_slope",
]
x_list += ["psd_lfp_csd"]


## %% Creates the design matrix and solve the linear system
import scipy.sparse as sp
from iblutil.numerical import ismember

y_name = "cosmos_id"
x = df_voltage.loc[:, x_list].values
nc, _ = df_voltage.shape
aids = np.unique(df_voltage[y_name])
pids = np.unique(df_voltage.index.get_level_values(0))

_, ipids = ismember(df_voltage.index.get_level_values(0), pids)
_, iregions = ismember(df_voltage[y_name], aids)

design_matrix_pids = sp.coo_matrix(
    (np.ones(nc), (np.arange(nc), ipids)), shape=(nc, pids.size)
)
design_matrix_regions = sp.coo_matrix(
    (np.ones(nc), (np.arange(nc), iregions)), shape=(nc, aids.size)
)

design_matrix = sp.hstack((design_matrix_pids, design_matrix_regions))
regions_probes = np.zeros((pids.size + aids.size, x.shape[1]))

# Creates the target vector
for ix in np.arange(x.shape[1]):
    print(f"Processing {ix} out of {x.shape[1]}")
    regions_probes[:, ix] = sp.linalg.lsqr(design_matrix, x[:, ix])[0]

x_debias = x - regions_probes[: pids.shape[0], :][ipids, :]
# Convert back into dataframe and save
df_voltage_debias = pd.DataFrame(data=x_debias, index=df_voltage.index, columns=x_list)
df_voltage_debias[y_name] = df_voltage[y_name]
# df_voltage_debias.to_parquet(local_path.joinpath('df_voltage_debias.pqt'))


##


# Try encoder models
import ephys_atlas.encoding
from sklearn.metrics import r2_score

train_idx, test_idx = ephys_atlas.encoding.train_test_split_indices(
    df_voltage, include_benchmarks=True
)

null_model = ephys_atlas.encoding.NullModel01()
null_model.fit(df_voltage.loc[train_idx, :], x_list=x_list, y_name=y_name)

debias_model = ephys_atlas.encoding.NullModel01()
debias_model.fit(df_voltage_debias.loc[train_idx, :], x_list=x_list, y_name=y_name)

null_pred, pred_idx = null_model.predict(df_voltage.loc[test_idx, y_name])
debias_xpred, pred_idx = debias_model.predict(df_voltage_debias.loc[test_idx, y_name])

for feature in x_list:
    null = r2_score(
        df_voltage.loc[test_idx, feature].values[pred_idx],
        null_pred.loc[:, feature].values,
    )
    debias = r2_score(
        df_voltage_debias.loc[test_idx, feature].values[pred_idx],
        debias_xpred.loc[:, feature].values,
    )
    print(f"{null: .3f}, {debias: .3f}, {feature}")

##


## %% Train decoder
from sklearn.ensemble import RandomForestClassifier

kwargs = {
    "n_estimators": 30,
    "max_depth": 25,
    "max_leaf_nodes": 10000,
    "random_state": 420,
    "n_jobs": -1,
    "criterion": "entropy",
}


def train_decoder(x):
    test_idx = df_voltage.index.get_level_values(0).isin(benchmark_pids)
    train_idx = ~test_idx

    x_train = x[train_idx, :]
    x_test = x[test_idx, :]
    for mapping in ["cosmos_id", "beryl_id", "atlas_id"]:
        y_test = df_voltage.loc[test_idx, mapping]
        y_train = df_voltage.loc[train_idx, mapping]
        classifier = RandomForestClassifier(verbose=True, **kwargs)
        clf = classifier.fit(x_train, y_train)
        y_null = np.random.choice(df_voltage.loc[train_idx, mapping], y_test.size)
        print(mapping, clf.score(x_test, y_test), clf.score(x_test, y_null))
        break


train_decoder(x)
train_decoder(x_debias)

## %% Unsupervised classifier
import cebra
import torch
from sklearn.preprocessing import StandardScaler

mapping = "cosmos_id"


def train_cebra(x):
    torch.cuda.empty_cache()
    # print(cebra.models.get_options('*', limit = 40))
    model = cebra.CEBRA(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=0.0003,  # 0.001
        temperature=1,
        output_dimension=4,
        max_iterations=10000,
        distance="cosine",
        conditional="time_delta",
        device="cuda",
        verbose=True,
        time_offsets=32,
    )

    SCALE = True
    if SCALE:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
    model.fit(x)
    cebra.plot_loss(model)
    embedding = model.transform(x)
    return model, embedding


y = df_voltage.loc[:, mapping]
model_baseline, embedding_baseline = train_cebra(df_voltage.loc[:, x_list])
# model_sc, embedding_sc = train_cebra(x_debias)
from iblutil.numerical import ismember

mapping = "atlas_id"  #  'cosmos_id', 'atlas_id']:
rgb = regions.rgb[ismember(y, regions.id)[1]].astype(np.float32) / 255

import matplotlib.pyplot as plt

plt.style.use("dark_background")
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].scatter(
    embedding_baseline[:, 0], embedding_baseline[:, 1], s=2, c=rgb, alpha=0.05
)
ax[0].set_axis_off()


## %%

import numpy as np
import datoviz

y = df_voltage.loc[:, "atlas_id"]
color = regions.rgba[ismember(y, regions.id)[1]]
color[:, -1] = 40
ms = np.ones_like(y) * 3
# We create a scene with one row and two columns.
c = datoviz.canvas(show_fps=True)
s = c.scene(1, 1)

# We add the two panels with different controllers.
panel0 = s.panel(0, 0, controller="arcball")

# We create a visual in each panel.
visual = panel0.visual("point")
visual.data("pos", embedding_baseline[:, :3])
visual.data("color", color)
visual.data("ms", ms)
datoviz.run()

# c.screenshot('/datadisk/Data/paper-ephys-atlas/toto.png')
