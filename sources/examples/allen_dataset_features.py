import h5py as h5
import numpy as np

import slidingRP.metrics
import brainbox.metrics.single_units

spikes_nwb_file = (
    "/home/ibladmin/Documents/olivier/2022-10_CSD/data/mouse405751.spikes.nwb"
)
nwb = h5.File(spikes_nwb_file, "r")

## nwb.visit(print)

results = {}
for probe in nwb["processing"].keys():
    st = nwb["processing"][probe]["Clustering"]["times"][()]
    sc = nwb["processing"][probe]["Clustering"]["num"][()]
    sa = nwb["processing"][probe]["Clustering"]["peak_over_rms"][()]
    cdx = nwb["processing"][probe]["Clustering"]["cluster_nums"][()]

    srp = slidingRP.metrics.slidingRP_all(
        spikeTimes=st,
        spikeClusters=sc,
        **{"sampleRate": 30000, "binSizeCorr": 1 / 30000},
    )
    slidingRP_viol = srp["value"]

    noise_cutoff = np.zeros(cdx.size)
    amp_median = np.zeros(cdx.size)
    params = brainbox.metrics.single_units.METRICS_PARAMS
    for ic in np.arange(cdx.size):
        sel = sc == cdx[ic]
        _, noise_cutoff[ic], _ = brainbox.metrics.single_units.noise_cutoff(
            sa[sel], **params["noise_cutoff"]
        )
        amp_median[ic] = np.median(sa[sel])

    labels = np.c_[
        np.array(slidingRP_viol),
        noise_cutoff < params["noise_cutoff"]["nc_threshold"],
        amp_median > params["med_amp_thresh_uv"] / 1e6,
    ]
    ngood = np.sum(np.sum(labels, axis=-1) == 3)

    clusters = dict(ind=nwb["processing"][probe]["unit_list"][()])
    # clusters['label'] = np.zeros_like(clusters['ind'])
    # for ic in clusters['ind']:
    #     nwb['processing'][probe]['UnitTimes'][str(ic)].keys()
    #     print(ic)

    results[probe] = dict(n_ibl=ngood, n=cdx.size, n_allen=clusters["ind"].size)
    break

import pandas as pd

res = pd.DataFrame(results).T
res["p_allen"] = res["n_allen"] / res["n"] * 100
res["p_ibl"] = res["n_ibl"] / res["n"] * 100

#         n_ibl    n  n_allen    p_allen      p_ibl
# probeA     58  304      250  82.236842  19.078947
# probeB     60  300      231  77.000000  20.000000
# probeC     94  508      379  74.606299  18.503937
# probeD     31  362      203  56.077348   8.563536
# probeE     74  385      301  78.181818  19.220779
# probeF     45  268      182  67.910448  16.791045
