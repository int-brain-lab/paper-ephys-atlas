from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from ephys_atlas.anatomy import EncodingAtlas

OUT_PATH = Path('/Users/olivier/Library/CloudStorage/GoogleDrive-olivier.winter@internationalbrainlab.org/Shared drives/Task Force - Electrophysiology Atlas/Decoding/NEMO')
OUT_PATH = Path("/datadisk/team_drives/Task Force - Electrophysiology Atlas/Decoding/NEMO")

ba = EncodingAtlas()
cosmos_aids = np.sort(ba.regions.id[np.unique(ba.regions.mappings['Cosmos'])])
cosmos_volume = ba.regions.mappings['Cosmos'][ba.label]
nr = cosmos_aids.size
# 4tsUsGkNC5vMcoNgGF
# %%
SIGMA = 2
soft_boundaries = np.zeros((ba.label.size, nr), dtype=np.float16)

for i, aid in enumerate(cosmos_aids):
    rid = np.where(ba.regions.id == aid)[0][0]
    print(rid, aid, ba.regions.name[rid])
    soft_boundaries[:, i] = gaussian_filter(
        (cosmos_volume == rid).astype(np.float32), SIGMA).astype(np.float16).flatten()
    print(np.isnan(soft_boundaries[:, i]).sum())
soft_boundaries = soft_boundaries / np.sum(soft_boundaries, axis=1)[:, np.newaxis]
assert np.isnan(soft_boundaries).sum() == 0

np.savez(OUT_PATH.joinpath('soft_boundaries.npz'),
         soft_boundaries=soft_boundaries,
         cosmos_aids=cosmos_aids)

# %%
# x = np.zeros(300)
# x[150] = 1
# plt.plot(gaussian_filter(x, 10))
sig = 2
x = np.linspace(-1000, 1000, 100)
x = np.arange(-250, 250, 25)

g = 1 / np.sqrt(2 * np.pi * sig) * np.exp( - x ** 2 / (2 * sig ** 2))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.figure()
plt.plot(x * 25, g)


b = np.zeros(x.size)
b[:int(x.size / 2)] = 1
bb = np.zeros((x.size, 2))
for i in range(2):
    bb[:, i] = gaussian_filter((b == i).astype(np.float32), 2)

bb = bb / np.sum(bb, axis=1)[:, np.newaxis]
bound = (b.sum() - .5) * 25
fig, ax = plt.subplots()
ax.plot(x, bb)
ax.add_patch(Rectangle((x[0], 0), width=bound, height=1, facecolor='orange', alpha=0.3))
ax.add_patch(Rectangle((bound + x[0], 0), width=250 * 2 - bound, height=1, facecolor='blue', alpha=0.3))
ax.set(xlabel='depth (um)')
