import pandas as pd

from pathlib import Path
from one.remote import aws
from one.api import ONE

LOCAL_DATA_PATH = Path("/Users/olivier/Documents/datadisk/atlas")
LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/features_tables")
# The AWS private credentials are stored in Alyx, so that only one authentication is required
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
aws.s3_download_folder("aggregates/bwm",
                       LOCAL_DATA_PATH,
                       s3=s3, bucket_name=bucket_name)

df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))
df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('raw_ephys_features.pqt'))

## %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=df_probes, x='histology', palette='deep')

plt.figure()
sns.countplot(data=df_clusters, x='label',  palette='deep')


