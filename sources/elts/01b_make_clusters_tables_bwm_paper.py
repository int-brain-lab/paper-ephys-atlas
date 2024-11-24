from pathlib import Path
import numpy as np
import pandas as pd
import urllib.error
from datetime import date

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblutil.util import setup_logger

from neuropixel import trace_header
from iblutil.numerical import ismember2d
from brainwidemap import bwm_query
from brainbox.io.one import SpikeSortingLoader

logger = setup_logger("brainbox")

one = ONE()
ba = AllenAtlas()

year_week = date.today().isocalendar()[:2]
STAGING_PATH = Path("/mnt/s0/aggregates/___2022_Q4_IBL_et_al_BWM").joinpath(
    f"{year_week[0]}_W{year_week[1]:02}_bwm"
)
CACHE_DIR = Path(
    "/mnt/s1/bwm"
)  # this is the path containing the metrics and clusters tables for fast releoading

excludes = []
errorkey = []
error404 = []
one = ONE(base_url="https://alyx.internationalbrainlab.org")
# pids, _ = bwm_pids(one, tracing=True)
bwm_df = bwm_query(one)
pids = bwm_df["pid"]
# init dataframes
df_probes = pd.DataFrame(
    dict(eid="", pname="", spike_sorter="", histology=""), index=pids
)
ldf_channels = []
ldf_clusters = []
ldf_depths = []
no_spike_sorting = []

pids = [
    "7791ee46-5c13-4d1b-8518-5602dcb8666b",
    "aac3b928-e99a-4039-ace1-af45d0130d82",
    "19c5b0d5-a255-47ff-9f8d-639e634a7b61",
    "1ca6cd06-1ed5-45af-b73a-017d5e7cff48",
    "0259543e-1ca3-48e7-95c9-53f9e4c9bfcc",
    "e92f8734-2c06-4168-9271-d00b3bf57c02",
    "46cd9c0a-39de-4aeb-90a6-86a2fda0b1a4",
    "2cbb5bc7-edbd-431e-a931-21e466d20dec",
    "3c9c3757-32dd-40cf-83ec-5e21731ce9c5",
    "1878c999-d523-474a-9d4e-8dde53d7324c",
    "eb7e9f3f-b79d-4cdd-bc24-b13a4008c1b5",
    "aec2b14f-5dbc-400b-bf2e-dd13e711e2ff",
    "16799c7a-e395-435d-a4c4-a678007e1550",  # key error
]

## %%
IMIN = 0

for i, pid in enumerate(pids):
    if i < IMIN:
        continue
    if pid in excludes:
        continue
    eid, pname = one.pid2eid(pid)
    df_probes["eid"][i] = eid
    df_probes["pname"][i] = pname

    # spikes, clusters, channels = load_spike_sorting_fast(eid=eid, probe=pname, one=one, nested=False)
    logger.info(f"{i}/{len(pids)}, {pid}")
    ss = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    try:
        spikes, clusters, channels = ss.load_spike_sorting(query_type="remote")
    except urllib.error.HTTPError:
        error404.append(pid)
        logger.error(f"{pid} error 404")
        continue
    except KeyError:
        errorkey.append(pid)
        logger.error(f"{pid} key error")
        continue
    cache_dir_cluster = CACHE_DIR.joinpath(f"{pid}")
    cache_dir_cluster.mkdir(exist_ok=True)
    if cache_dir_cluster.joinpath("clusters.pqt").exists():
        df_clusters = pd.read_parquet(cache_dir_cluster.joinpath("clusters.pqt"))
        _clusters = {}
        for k in df_clusters.keys():
            _clusters[k] = df_clusters[k].values
            # if k in ['uuids', 'acronym']:
            #     assert np.all(_clusters[k] == clusters[k])
            # else:
            #     assert np.all(np.isclose(_clusters[k], clusters[k], equal_nan=True))
        clusters = _clusters
    else:
        clusters = ss.merge_clusters(
            spikes,
            clusters,
            channels,
            compute_metrics=True,
            cache_dir=cache_dir_cluster,
        )
    df_probes["spike_sorter"][i] = ss.collection
    df_probes["histology"][i] = ss.histology
    if not spikes:
        no_spike_sorting.append(pid)
        continue
    df_ch = pd.DataFrame(channels)
    df_ch["pid"] = pid
    ldf_channels.append(df_ch)
    df_clu = pd.DataFrame(clusters)
    df_clu["pid"] = pid
    ldf_clusters.append(df_clu)
    # aggregate spike features per depth
    df_spikes = pd.DataFrame(spikes)
    df_spikes.dropna(axis=0, how="any", inplace=True)
    df_spikes["rdepths"] = (np.round(df_spikes["depths"] / 20) * 20).astype(np.int32)
    df_spikes["amps"] = df_spikes["amps"] * 1e6
    df_depths = df_spikes.groupby("rdepths").agg(
        amps=pd.NamedAgg(column="amps", aggfunc="median"),
        amps_std=pd.NamedAgg(column="amps", aggfunc="std"),
        cell_count=pd.NamedAgg(column="clusters", aggfunc="nunique"),
        spike_rate=pd.NamedAgg(column="amps", aggfunc="count"),
    )
    df_depths["pid"] = pid
    df_depths["spike_rate"] = df_depths["spike_rate"] / (
        np.max(spikes["times"]) - np.min(spikes["times"])
    )
    ldf_depths.append(df_depths)


## %%
df_channels = pd.concat(ldf_channels, ignore_index=True)
df_clusters = pd.concat(ldf_clusters, ignore_index=True)
df_depths = pd.concat(ldf_depths)

# convert the channels dataframe to a multi-index dataframe
h = trace_header(version=1)
_, chind = ismember2d(
    df_channels.loc[:, ["lateral_um", "axial_um"]].to_numpy(), np.c_[h["x"], h["y"]]
)
df_channels["raw_ind"] = chind
df_channels = df_channels.set_index(["pid", "raw_ind"])

# convert the depths dataframe to a multi-index dataframe
df_depths["depths"] = df_depths.index.values
df_depths = df_depths.set_index(["pid", "depths"])

# saves the 3 dataframes
STAGING_PATH.mkdir(exist_ok=True, parents=True)
df_channels.to_parquet(STAGING_PATH.joinpath("channels.pqt"))
df_clusters.to_parquet(STAGING_PATH.joinpath("clusters.pqt"))
df_probes.to_parquet(STAGING_PATH.joinpath("probes.pqt"))
df_depths.to_parquet(STAGING_PATH.joinpath("depths.pqt"))


print(f'aws s3 sync "{STAGING_PATH}" s3://ibl-brain-wide-map-private/aggregates/bwm')
print(errorkey)
print(error404)

# Run August 22nd 2022
# In [35]: error404
# Out[35]:
# ['b2ea68e2-c732-4d17-8166-1a8595fff225',
#  '577e4741-4b15-4e91-b81b-61304a09bfb5',
#  '9b3ad89a-177f-4242-9a96-2fd98721e47f',
#  '367ea4c4-d1b7-47a6-9f18-5d0df9a3f4de',
#  '4279e354-a6b8-4eff-8245-7c8723b07834',
#  '507b20cf-4ab0-4f55-9a81-88b02839d127',
#  'f2a098e7-a67e-4125-92d8-36fc6b606c45',
#  'f967a527-257f-404a-871d-b91575dca3b4',
#  'ffb1b072-2de7-44a4-8115-5799b9866382',
#  'de5d704c-4a5b-4fb0-b4af-71614c510a8b',
#  '2201eb05-aebe-4bda-905e-b5cc1baf3840',
#  '39180bcb-13e5-46f9-89e1-8ea2cba22105',
#  '04db6b9e-a80c-4507-a98e-ad76294ac444',
#  '531423f6-d36d-472b-8234-c8f7b8293f79']
