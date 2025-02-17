'''
Prepare the dataset of features for the subject with epileptic seizure
There are 2 dataframes:
One dataframe is extracted at a time of normal brain activity,
the other is extraction at a time of seizure
'''
# Get the dataframes from Google drive:
# https://drive.google.com/drive/u/0/folders/1Qvj16dii3g1myCm40gvoxZ3Ih7tjeBN-

import pandas as pd
from pathlib import Path
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

one = ONE()

folder_seizure = Path('/Users/gaellechapuis/Documents/Work/EphysAtlas/seizure')

df_baseline=pd.read_parquet(folder_seizure.joinpath('5246af08.pqt'))
df_seizure=pd.read_parquet(folder_seizure.joinpath('5246af08-seizure.pqt'))

# Add channel information to the dataframes

pid = "5246af08-0730-40f7-83de-29b5d62b9b6d"
ssl = SpikeSortingLoader(pid=pid, one=one)
channels = ssl.load_channels()

# Create dataframe from channels
df_ch = pd.DataFrame(channels)

# Add each column to the dataframe
df_baseline = pd.merge(df_baseline, df_ch, on=['lateral_um', 'axial_um'], how='left')
df_seizure = pd.merge(df_seizure, df_ch, on=['lateral_um', 'axial_um'], how='left')

# Save
df_baseline.to_parquet(folder_seizure.joinpath('col_5246af08.pqt'))
df_seizure.to_parquet(folder_seizure.joinpath('col_5246af08-seizure.pqt'))
