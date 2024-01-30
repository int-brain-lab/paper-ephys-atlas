"""
Example to download histology registered electrophysiology features tables
 and cluster tables for ephys features analytics.

see https://docs.google.com/document/d/1_B-h9YHKmM5ggd5pA_qhWr8MQdgnX2PuZyidDdvauZk/edit for a full
description of the data structure.
"""

from pathlib import Path

from one.api import ONE
import ephys_atlas.data

LABEL = '2024_W04'  # 2023_W51_autism for example would return the autism dataset features
LOCAL_DATA_PATH = Path('/home/ibladmin/scratch/')
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(
    label=LABEL, local_path=LOCAL_DATA_PATH, one=one)
# df_raw_features, df_clusters, df_channels = load_tables(local_path=FOLDER_GDRIVE)

# df_depths = ephys_atlas.data.compute_depth_dataframe(df_raw_features, df_clusters, df_channels)
df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)


## %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')
