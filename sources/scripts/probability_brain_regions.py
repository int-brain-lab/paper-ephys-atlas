"""
To reflect the uncertainty of the localisation, one can lookup a volume around
a given coordinate and aggregate each brain region by proportion. This is done
by specifying a "radius_um" parameter in the Atlas lookup function.
"""
import numpy as np
from ibllib.atlas import AllenAtlas

ba = AllenAtlas()
xyz = np.array([0, -.0024, -.0038])
aid = ba.get_labels(xyz, mapping='Beryl')
aids, probabilities = ba.get_labels(xyz, mapping='Beryl', radius_um=250)

## %% Display this on a slice
from iblutil.numerical import ismember

axs = ba.plot_slices(np.array([0, -.0024, -.0038]))

axs[1, 1].bar(np.arange(len(aids)), probabilities,
              tick_label=ba.regions.id2acronym(aids, mapping='Beryl'),
              color=ba.regions.rgb[ ismember(aids, ba.regions.id)[1]] / 255,
              edgecolor='k',
              )