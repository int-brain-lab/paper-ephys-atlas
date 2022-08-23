import pandas as pd

from pathlib import Path
from one.remote import aws
from one.api import ONE

LOCAL_DATA_PATH = Path("/Users/olivier/Documents/datadisk/atlas")

# The AWS private credentials are stored in Alyx, so that only one authentication is required
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='online')
s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
aws.s3_download_folder("data/tables/atlas",
                       LOCAL_DATA_PATH,
                       s3=s3, bucket_name=bucket_name)

clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))


## %%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
sns.countplot(data=probes, x='histology', palette='deep')

plt.figure()
sns.countplot(data=clusters, x='label',  palette='deep')

## %%

