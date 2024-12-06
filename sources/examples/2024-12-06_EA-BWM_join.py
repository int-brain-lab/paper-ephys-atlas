from pathlib import Path

from one.api import ONE
from one import __version__
from brainwidemap.bwm_loading import bwm_units, load_good_units
import ephys_atlas.data

LOCAL_DATA_PATH = Path('/mnt/s0/ephys-atlas-decoding')
print(f"one version {__version__}")
one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')

df_voltage, _, _, df_probes = ephys_atlas.data.load_voltage_features(LOCAL_DATA_PATH.joinpath('features', '2024_W04'))

units_df = bwm_units(one)

pids_bwm = units_df['pid'].unique()
pids_atlas = df_voltage['pids'].unique()

len(set(pids_bwm) - set(pids_atlas))  # we have 36 pids not in the atlas

# this dataframes contains the ephys features for the BWM clusters.
df = units_df.merge(df_voltage, left_on=['pid', 'axial_um', 'lateral_um'], right_on=['pids', 'axial_um', 'lateral_um'], how='inner')
