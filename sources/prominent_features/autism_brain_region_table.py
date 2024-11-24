from pathlib import Path
import pandas as pd

from iblatlas.atlas import BrainRegions
from ephys_atlas.data import load_tables, download_tables
from one.api import ONE

onen = ONE()
br = BrainRegions()

label = "2023_W51_autism"
local_data_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Data")
force_download = False
mapping = "Beryl"

local_data_path_clusters = local_data_path.joinpath(label).joinpath("clusters.pqt")
if not local_data_path_clusters.exists() or force_download:
    print("Downloading table")
    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="local")
    df_voltage, df_clusters, df_channels, df_probes = download_tables(
        label=label, local_path=local_data_path, one=one
    )
else:
    df_voltage, df_clusters, df_channels, df_probes = load_tables(
        local_data_path.joinpath(label), verify=True
    )

df_voltage = pd.merge(
    df_voltage, df_channels, left_index=True, right_index=True
).dropna()
df_voltage = df_voltage.rename(
    columns={"atlas_id": "Allen_id", "acronym": "Allen_acronym"}
)

df_voltage["Cosmos_id"] = br.remap(
    df_voltage["Allen_id"], source_map="Allen", target_map="Cosmos"
)
df_voltage["Beryl_id"] = br.remap(
    df_voltage["Allen_id"], source_map="Allen", target_map="Beryl"
)
df_voltage["pids"] = df_voltage.index.get_level_values(0)
# Remove void / root
df_voltage.drop(
    df_voltage[df_voltage["Allen_acronym"].isin(["void", "root"])].index, inplace=True
)


##
# Compute region dataframe
df_regions = df_voltage.groupby(mapping + "_id").agg(
    n_channels=pd.NamedAgg(column=mapping + "_id", aggfunc="count"),
    n_pids=pd.NamedAgg(column="pids", aggfunc="nunique"),
)
df_regions[mapping + "_id"] = df_regions.index.values
df_regions[mapping + "_acronym"] = br.id2acronym(
    df_regions.index.values, mapping=mapping
)

df_regions = df_regions.sort_values("n_channels", ascending=False)
