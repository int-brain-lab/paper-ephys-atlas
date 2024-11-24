# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:10:12 2022

@author: guido
"""

import pandas as pd
from one.api import ONE

one = ONE()

# Load in data
chan_volt = pd.read_parquet(
    "C:\\Users\\guido\\Data\\EphysAtlas\\channels_voltage_features.pqt"
)
all_pids = chan_volt.index.get_level_values(level="pid").unique()

mm_coord = pd.DataFrame()
for i, pid in enumerate(all_pids):
    print(f"Insertion {i} of {len(all_pids)}")
    sess_info = one.alyx.rest("insertions", "list", id=pid)
    traj = one.alyx.rest(
        "trajectories",
        "list",
        session=sess_info[0]["session"],
        probe=sess_info[0]["name"],
        provenance="Micro-manipulator",
    )
    if len(traj) != 1:
        continue
    mm_coord = pd.concat(
        (
            mm_coord,
            pd.DataFrame(
                index=[pid],
                data={
                    "x": traj[0]["x"],
                    "y": traj[0]["y"],
                    "depth": traj[0]["depth"],
                    "theta": traj[0]["theta"],
                    "phi": traj[0]["phi"],
                },
            ),
        )
    )

# Save to disk
mm_coord.to_parquet("C:\\Users\\guido\\Data\\EphysAtlas\\coordinates.pqt")
