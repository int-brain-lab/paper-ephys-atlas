from pathlib import Path

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iblatlas.atlas import AllenAtlas

from ephys_atlas.encoding import NullModel01

ba = AllenAtlas()
mod = NullModel01(ba=ba)


SEED = 42
FRAC = 0.05
STAGING_PATH = Path("/datadisk/FlatIron/tables/bwm")

df_voltage = pd.read_parquet(STAGING_PATH.joinpath("channels_voltage_features.pqt"))
df_voltage["atlas_id_beryl"] = ba.regions.remap(
    df_voltage["atlas_id"].to_numpy(), target_map="Beryl"
)

df_train, df_test = mod.split_dataframe(df_voltage)

mod.fit(df_train)
firing_rate = mod.predict(df_test)
