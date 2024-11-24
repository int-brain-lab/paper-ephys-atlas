"""
Here we compute the state transition matrices from the ALlen Atlas,
for several mappings (Allen, Cosmos, Beryl)
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

from iblatlas.atlas import AllenAtlas
from iblutil.numerical import ismember

# from iblapps.atlasview import atlasview
# av = atlasview.view()
LOCAL_DATA_PATH = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-decoding")

# instantiate the Allen atlas
ba = AllenAtlas(res_um=25)
ba.compute_surface()

str_mapping = "Cosmos"
str_mapping = "Beryl"
# str_mapping = 'Allen'
volume = ba.regions.mappings[str_mapping][ba.label]  # ap, ml, dv
mask = ba.mask()
volume[~mask] = -1

# getting the unique set of regions for the given mapping
aids_unique = np.unique(ba.regions.id[ba.regions.mappings[str_mapping]])
_, ir_unique = ismember(aids_unique, ba.regions.id)

up = volume[:, :, :-1].flatten()
lo = volume[:, :, 1:].flatten()
iok = np.logical_and(up >= 0, lo >= 0)
_, icc_up = ismember(up[iok], ir_unique)
_, icc_lo = ismember(lo[iok], ir_unique)

# here we count the number of voxel from each reagion
state_transitions = sp.coo_matrix(  # (data, (i, j))
    (np.ones_like(icc_lo), (icc_up, icc_lo)), shape=(ir_unique.size, ir_unique.size)
).todense()
states_count = sp.coo_matrix(
    (np.ones_like(icc_lo), (icc_up, icc_up * 0)), shape=(ir_unique.size, 1)
).todense()

up2down = state_transitions / state_transitions.sum(axis=1)

np.savez(
    LOCAL_DATA_PATH.joinpath(f"region_transition_{str_mapping.lower()}.npz"),
    region_transitions=state_transitions,
    region_counts=states_count.squeeze(),
    region_aids=ba.regions.id[ir_unique],
)


## %%

fig, ax = plt.subplots(figsize=(12, 10))
if str_mapping == "Cosmos":
    sns.heatmap(
        up2down * 100,
        vmin=0,
        vmax=1.2,
        annot=ir_unique.size < 50,
        fmt=".1f",
        ax=ax,
        cmap="Blues",
    )
    ax.set(
        xticklabels=ba.regions.acronym[ir_unique],
        yticklabels=ba.regions.acronym[ir_unique],
    )
else:
    sns.heatmap(
        up2down * 100,
        vmin=0,
        vmax=1.2,
        annot=ir_unique.size < 50,
        fmt=".1f",
        ax=ax,
        cmap="Blues",
    )

ax.set(
    title=f"Probability of next channel switching {str_mapping} region (%)",
    ylabel="Current channel",
    xlabel="Next channel down",
)


## %%
