import ephys_atlas.workflow as workflow
import time
import joblib

from pathlib import Path
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import download_tables, load_voltage_features
from ephys_atlas.encoding import voltage_features_set
from one.api import ONE
from ephys_atlas.entropy import compute_info_gain

##
# Download voltage and precise mapping

one = ONE()
br = BrainRegions()

label = '2023_W51'  # label = '2023_W51_autism'
mapping = 'Cosmos'
# local_data_path = Path('/mnt/s0/aggregates/')  # TODO ASK OW
# save_folder = Path('/mnt/s0/ephys-atlas-decoding/entropy/')  # TODO ASK OW
local_data_path = Path('/Users/gaelle/Documents/Work/EphysAtlas/Data')
save_folder = Path('/Users/gaelle/Documents/Work/EphysAtlas/Entropy_DF_WF')
if not save_folder.exists():
    save_folder.mkdir()

force_download = False

local_data_path_clusters = local_data_path.joinpath(label).joinpath('clusters.pqt')
if not local_data_path_clusters.exists() or force_download:
    print('Downloading table')
    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
    _, _, _, _ = download_tables(
        label=label, local_path=local_data_path, one=one)
# Re-load to make sure all columns have the proper nomenclature
df_voltage, df_clusters, df_channels, df_probes = load_voltage_features(local_data_path.joinpath(label))

# Remove void / root
df_voltage.drop(df_voltage[df_voltage[mapping+'_acronym'].isin(['void', 'root'])].index, inplace=True)
# Get whole set of features
features = voltage_features_set()

##
t = time.time()
joblib.Parallel(n_jobs=5)(joblib.delayed(compute_info_gain)(df_voltage, feature, mapping, save_folder) for feature in features)
print(time.time() - t, len(features), mapping)
