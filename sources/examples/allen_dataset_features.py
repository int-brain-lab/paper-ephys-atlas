import h5py as h5
import numpy as np

import slidingRP.metrics
import brainbox.metrics.single_units

spikes_nwb_file = "/home/ibladmin/Documents/olivier/2022-10_CSD/data/mouse405751.spikes.nwb"
nwb = h5.File(spikes_nwb_file, 'r')

## nwb.visit(print)

for probe in nwb['processing'].keys():
    st = nwb['processing'][probe]['Clustering']['times'][()]
    sc = nwb['processing'][probe]['Clustering']['num'][()]
    sa = nwb['processing'][probe]['Clustering']['peak_over_rms'][()]
    cdx = nwb['processing'][probe]['Clustering']['cluster_nums'][()]


    srp = slidingRP.metrics.slidingRP_all(spikeTimes=st, spikeClusters=sc,
                                          **{'sampleRate': 30000, 'binSizeCorr': 1 / 30000})
    slidingRP_viol = srp['value']

    noise_cutoff = np.zeros(cdx.size)
    amp_median = np.zeros(cdx.size)
    params = brainbox.metrics.single_units.METRICS_PARAMS
    for ic in np.arange(cdx.size):
        sel = sc == cdx[ic]
        _, noise_cutoff[ic], _ = brainbox.metrics.single_units.noise_cutoff(sa[sel], **params['noise_cutoff'])
        amp_median[ic] = np.median(sa[sel])


    labels = np.c_[
        np.array(slidingRP_viol),
        noise_cutoff < params['noise_cutoff']['nc_threshold'],
        amp_median > params['med_amp_thresh_uv'] / 1e6,
    ]
    ngood = np.sum(np.sum(labels, axis=-1) == 3)
    print(probe,  ngood, cdx.size, ngood / cdx.size * 100)

# probeA 58 304 19.078947368421055
# probeB 60 300 20.0
# probeC 94 508 18.503937007874015
# probeD 31 362 8.56353591160221
# probeE 74 385 19.22077922077922
# probeF 45 268 16.791044776119403
