from pathlib import Path

import pandas as pd
import numpy as np
import cebra
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from iblutil.numerical import ismember
import ephys_atlas.data
import ephys_atlas.plots
from iblatlas.regions import BrainRegions
from one.api import ONE

one = ONE(mode='remote')
regions = BrainRegions()
local_path = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
# local_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Data/")
local_path = Path("/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")
label = '2024_W04'
# df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(local_path, label=label, one=one, verify=True)
df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_voltage_features(local_path.joinpath(label))

x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'spike_count', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']
x_list += ['peak_time_secs', 'peak_val', 'trough_time_secs', 'trough_val', 'tip_time_secs', 'tip_val']
x_list += ['polarity', 'depolarisation_slope', 'repolarisation_slope', 'recovery_time_secs', 'recovery_slope']
x_list += ['psd_lfp_csd']

mapping = 'Cosmos_id'

def train_cebra(x, device='cuda', y=None):
    torch.cuda.empty_cache()
    # print(cebra.models.get_options('*', limit = 40))
    model = cebra.CEBRA(
        model_architecture='offset10-model',
        batch_size=512,
        learning_rate=0.0003,  # 0.001
        temperature=1,
        output_dimension=4,
        max_iterations=10000,
        distance='cosine',
        conditional='time_delta',
        device=device,
        verbose=True,
        time_offsets=32
    )

    SCALE = True
    if SCALE:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
    if y is None:
        model.fit(x)
    else:
        model.fit(x, y)
    cebra.plot_loss(model)
    embedding = model.transform(x)
    return model, embedding

y = df_voltage.loc[:, mapping]
model_baseline, embedding_baseline = train_cebra(df_voltage.loc[:, x_list], device='cpu')
model_baseline, embedding_baseline = train_cebra(df_voltage.loc[:, x_list], device='cpu', y=y)

rgb = regions.rgb[ismember(y, regions.id)[1]].astype(np.float32) / 255

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].scatter(embedding_baseline[:, 0], embedding_baseline[:, 1], s=2, c=rgb, alpha=.05)
ax[0].set_axis_off()


## %%

import numpy as np
import datoviz
y = df_voltage.loc[:, 'atlas_id']
color = regions.rgba[ismember(y, regions.id)[1]]
color[:, -1] = 40
ms = np.ones_like(y) * 3
# We create a scene with one row and two columns.
c = datoviz.canvas(show_fps=True)
s = c.scene(1, 1)

# We add the two panels with different controllers.
panel0 = s.panel(0, 0, controller='arcball')

# We create a visual in each panel.
visual = panel0.visual('point')
visual.data('pos', embedding_baseline[:, :3])
visual.data('color', color)
visual.data('ms', ms)
datoviz.run()

# c.screenshot('/datadisk/Data/paper-ephys-atlas/toto.png')
