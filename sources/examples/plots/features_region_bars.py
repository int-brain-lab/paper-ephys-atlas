from pathlib import Path

import pandas as pd

# from brainwidemap.meta import meta_bwm
import ephys_atlas.data
import ephys_atlas.plots
from iblatlas.atlas import BrainRegions

br = BrainRegions()
# local_path = Path("/Users/olivier/Documents/datadisk/paper-ephys-atlas/ephys-atlas-decoding/latest")
local_data_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Data")
label = "2023_W41"
local_path = local_data_path.joinpath(label)
df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.load_tables(
    local_path, verify=True
)
df = df_voltage.merge(df_channels, left_index=True, right_index=True)

# dfa, palette = meta_bwm.get_allen_info()
df["atlas_id_beryl"] = br.remap(df["atlas_id"], source_map="Allen", target_map="Beryl")
df["pids"] = df.index.get_level_values(0)
# Index(['alpha_mean', 'alpha_std', 'spike_count', 'cloud_x_std', 'cloud_y_std',
#        'cloud_z_std', 'peak_trace_idx', 'peak_time_idx', 'peak_val',
#        'trough_time_idx', 'trough_val', 'tip_time_idx', 'tip_val', 'rms_ap',
#        'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
#        'psd_gamma', 'x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um',
#        'lateral_um', 'histology', 'x_target', 'y_target', 'z_target',
#        'atlas_id_target'],

## %% aggregate ber brain region
df_regions = df.groupby("atlas_id_beryl").agg(
    n_channels=pd.NamedAgg(column="atlas_id", aggfunc="count"),
    n_pids=pd.NamedAgg(column="pids", aggfunc="nunique"),
)

df_regions["atlas_id_beryl"] = df_regions.index.values
df_regions["acronym_beryl"] = br.id2acronym(df_regions.index.values, mapping="Beryl")

df_regions = df_regions.set_index("acronym_beryl").sort_index()
feature = df_regions["n_pids"].values
atlas_id = df_regions["atlas_id_beryl"]
label = "N of insertions"

## %% arguments would be atlas_id, feature, label, br
fig, ax = ephys_atlas.plots.region_bars(
    atlas_id=atlas_id, feature=feature, label=label, regions=br
)
