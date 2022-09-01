"""
Meant to be run on parede
cd /home/ibladmin/Documents/PYTHON/SPIKE_SORTING
conda activate pyks2
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from spike_psvae.subtract import make_channel_index, subtract_and_localize_numpy
from neurodsp.utils import rms
import neuropixel
from ibllib.misc import check_nvidia_driver
from iblutil.util import get_logger
from neurodsp.utils import WindowGenerator

print(torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
check_nvidia_driver()

ROOT_PATH = Path("/mnt/s0/ephys-atlas")
# ROOT_PATH = Path("/datadisk/scratch/ephys-atlas")
logger = get_logger('ibllib', file=ROOT_PATH.joinpath('localisation_log.txt'))

h = neuropixel.trace_header(version=1)
geom = np.c_[h['x'], h['y']]
xy = h['x'] + 1j * h['y']

kwargs = dict(
    extract_radius=200.,
    loc_radius = 100.,
    dedup_spatial_radius = 70.,
    thresholds=[12, 10, 8, 6, 5],
    radial_parents=None,
    tpca=None,
    device=None,
    probe="np1",
    trough_offset=42,
    spike_length_samples=121,
    loc_workers=1
)
channel_index = make_channel_index(geom, kwargs['extract_radius'], distance_order=False)


all_files = list(ROOT_PATH.rglob('chunk*_destripe.npy'))
for i, file_destripe in enumerate(all_files):
    file_waveforms = Path(str(file_destripe).replace('_destripe.npy', '_waveforms.npy'))
    file_spikes = Path(str(file_destripe).replace('_destripe.npy', '_spikes.pqt'))
    if file_waveforms.exists() and file_spikes.exists():
        continue
    logger.info(f"{i}/{len(all_files)}: {file_destripe}")
    data = np.load(file_destripe).astype(np.float32)
    # here the normalisation is based off a single chunk, but should this be constant for the whole recording ?
    data = data / rms(data, axis=-1)[:, np.newaxis]
    wg = WindowGenerator(data.shape[-1], 30000, overlap=0)
    localisation = []
    try:
        for first, last in wg.firstlast:
            loc, wfs = subtract_and_localize_numpy(data[:, first:last].T, geom, **kwargs)
            cleaned_wfs = wfs if first == 0 else np.concatenate([cleaned_wfs, wfs], axis=0)
            localisation.append(loc)
    except TypeError as e:
        logger.error(f"type error: {file_destripe}")
        continue

    localisation = pd.concat(localisation)
    np.save(file_waveforms, cleaned_wfs)
    localisation.to_parquet(file_spikes)
