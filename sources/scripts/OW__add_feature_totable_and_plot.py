'''
0. Look at the features already existing https://drive.google.com/drive/u/1/folders/1y2uaiyYnWqVJqtqMv4FxxFQ_8LT2W6xs
1. Define compute function
2. Download needed data (run example on 1 insertion)
3. Launch computation in loop
4. Append to dataframe ; save to dataframe
5. Display
'''

from one.api import ONE
from ibllib.atlas import AllenAtlas
import pandas as pd
from pathlib import Path
from ibllib.atlas.flatmaps import plot_swanson
import matplotlib.pyplot as plt
import numpy as np
from ephys_atlas.data import bwm_pids
from brainbox.io.one import SpikeSortingLoader
import urllib.error


# ==== INIT

one = ONE()
ba = AllenAtlas()
STAGING_PATH = Path('/Users/gaelle/Downloads/bwm_sav/')
cmap = 'Blues'

excludes = [
    '316a733a-5358-4d1d-9f7f-179ba3c90adf'
]

error404 = []

# Load already existing DF

df_channels = pd.read_parquet(STAGING_PATH.joinpath('channels.pqt'))

# Get pids

# pids, _ = bwm_pids(one, tracing=True)
pids = df_channels.index.values.tolist()
pids = [item[0] for item in pids]

# test
# pids = [pids[0]]
# pids = ['94e948c1-f7be-4868-893a-f7cd2df3313e']

# ==== Step 1 : Define compute function
# Name of column in dataframe
k = 'fanofactor'


def fanofactor(n_ch):  # take spikes as input for example
    # n_ch = 384
    v = np.random.rand(1, n_ch)
    return v[0]


#  Add column to dataframe
df_channels[k] = np.nan

# ==== Step 2-3 : Download needed data and Launch computation in loop
for i, pid in enumerate(pids):
    # Load data
    ''' LOAD DATA SKIPPED FOR NOW
    eid, pname = one.pid2eid(pid)
    ss = SpikeSortingLoader(pid=pid, one=one, atlas=ba)

    try:
        spikes, clusters, channels = ss.load_spike_sorting()
    except urllib.error.HTTPError:
        error404.append(pid)
        continue
    '''
    # Compute and append to df
    n_ch = len(df_channels.loc[pid, k])
    df_channels.loc[pid, k] = fanofactor(n_ch=n_ch)

# ==== Step 4: save to dataframe
# df_channels.to_parquet(AGGREGATE_PATH.joinpath('channels.pqt'))

# ==== Aggregate per brain region to plot
df_regions = df_channels.groupby('atlas_id').agg({
    k: 'median'
})

feats = {
    k: dict(cmap=cmap)
}

# PLOT
kwargs = feats[k]

fig, ax = plt.subplots(figsize=(15, 8))
fig.tight_layout()
plot_swanson(acronyms=df_regions.index, values=df_regions[k],
             br=ba.regions, ax=ax, **kwargs)
