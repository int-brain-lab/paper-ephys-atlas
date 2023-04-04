from pathlib import Path
from one.api import ONE
from ephys_atlas.data import download_tables, compute_depth_dataframe
LABEL = '2022_W34'
LABEL = '2023_W14'
LOCAL_DATA_PATH = Path("/mnt/s1/ephys-atlas-decoding/tables")
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_raw_features, df_clusters, df_channels = download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)
df_depths = compute_depth_dataframe(df_raw_features, df_clusters, df_channels)
df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)

## %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')
