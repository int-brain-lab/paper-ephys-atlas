'''
0. Look at the features already existing https://drive.google.com/drive/u/1/folders/1y2uaiyYnWqVJqtqMv4FxxFQ_8LT2W6xs
1. Define compute function
2. Download needed data (run example on 1 insertion)
3. Launch computation in loop
4. Append to dataframe ; save to dataframe
5. Display
'''
from one.api import ONE
from ephys_atlas.data import bwm_pids
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
import urllib.error
import pandas as pd
from pathlib import Path


# ==== INIT

one = ONE()
ba = AllenAtlas()
STAGING_PATH = Path('/Users/gaelle/Downloads/bwm_sav/')

excludes = [
    'f86e9571-63ff-4116-9c40-aa44d57d2da9',  # 404 not found
    '16ad5eef-3fa6-4c75-9296-29bf40c5cfaa',  # 404 not found
    '511afaa5-fdc4-4166-b4c0-4629ec5e652e',  # 404 not found
    'f88d4dd4-ccd7-400e-9035-fa00be3bcfa8',  # 404 not found
]

error404 = []

pids, _ = bwm_pids(one, tracing=True)
# test
pids = [pids[0]]

# Load already existing DF

df_channels = pd.read_parquet(STAGING_PATH.joinpath('channels.pqt'))

# ==== Step 1 : Define compute function
# Name of column in dataframe
k = 'fanofactor'


def fanofactor():
    # bla


#  Add column to dataframe
df_channels[k] = 

# ==== Step 2-3 : Download needed data and Launch computation in loop
    for i, pid in enumerate(pids):
        eid, pname = one.pid2eid(pid)
        ss = SpikeSortingLoader(pid=pid, one=one, atlas=ba)

        try:
            spikes, clusters, channels = ss.load_spike_sorting()
        except urllib.error.HTTPError:
            error404.append(pid)
            continue

        df_channels.loc[pid, k] = fanofactor(pid)


# ==== Step 4: save to dataframe
df_channels.to_parquet(STAGING_PATH.joinpath('channels.pqt'))