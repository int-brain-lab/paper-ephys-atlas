"""
- Compute the ks-test by pair of region
- Save (flat) dataframe for each feature,
  containing the 2x region ID and the ks p-value and statistic
- Aggregate the feature dataframes into 1 for all features
"""

import time
import joblib
import dask.dataframe as dd
from pathlib import Path
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import download_tables, load_voltage_features
from ephys_atlas.encoding import voltage_features_set
from one.api import ONE
from ephys_atlas.entropy import compute_ks_test

# Download voltage and precise mapping
one = ONE()
br = BrainRegions()

label = "2023_W51"
mapping = "Beryl"
local_data_path = Path("/mnt/s0/aggregates/")
save_folder = Path(f"/mnt/s0/ephys-atlas-decoding/kstest/{label}/")

# local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
# save_folder = Path(f'/Users/gaelle/Documents/Work/EphysAtlas/Entropy_KS/{label}/')
if not save_folder.parent.exists():
    save_folder.parent.mkdir()
if not save_folder.exists():
    save_folder.mkdir()

force_download = False

local_data_path_clusters = local_data_path.joinpath(label).joinpath("clusters.pqt")
if not local_data_path_clusters.exists() or force_download:
    print("Downloading table")
    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode="local")
    _, _, _, _ = download_tables(label=label, local_path=local_data_path, one=one)
# Re-load to make sure all columns have the proper nomenclature
df_voltage, df_clusters, df_channels, df_probes = load_voltage_features(
    local_data_path.joinpath(label), mapping=mapping
)

# Remove void / root
df_voltage.drop(
    df_voltage[df_voltage[mapping + "_acronym"].isin(["void", "root"])].index,
    inplace=True,
)
# Get whole set of features
features = voltage_features_set()

##
# Compute all dataframes in batches
t = time.time()
joblib.Parallel(n_jobs=5)(
    joblib.delayed(compute_ks_test)(df_voltage, feature, mapping, save_folder)
    for feature in features
)
print(time.time() - t, len(features), mapping)

##
# Aggregate into 1 dataframe
df_info = dd.read_parquet(list(save_folder.rglob(f"*{mapping}__ks_test.pqt")))
df_info = df_info.compute()
# Save
df_info.to_parquet(save_folder.joinpath(f"{label}__{mapping}__overall_ks_test.pqt"))
