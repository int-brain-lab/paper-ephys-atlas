from pathlib import Path
from one.api import ONE
from ephys_atlas.data import download_tables
LABEL = '2022_W34'
LOCAL_DATA_PATH = Path("/mnt/s1/ephys-atlas-decoding/tables")
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')

df_voltage, df_clusters, df_channels = download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)


## %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')
