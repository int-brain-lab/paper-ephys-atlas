from one.api import ONE
from pathlib import Path
from one.api import ONE
import shutil
from ephys_atlas.data import atlas_pids
import ephys_atlas.rawephys
import pandas as pd
ROOT_PATH = Path("/mnt/s1/ephys-atlas")

pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
    '5f7766ce-8e2e-410c-9195-6bf089fea4fd',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]
import numpy as np
pid = pids[0]

from viewephys.gui import viewephys
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
from ibllib.atlas import BrainRegions

one = ONE(base_url="https://alyx.internationalbrainlab.org")
FS = 30000

regions = BrainRegions()

for pid in pids:
    path_pid = ROOT_PATH.joinpath(pid)
    for path_t0 in path_pid.glob('T*'):
        t0_str = path_t0.parts[-1]
        path_qc = path_pid.joinpath('pics')
        path_qc.mkdir(exist_ok=True)
        ap = np.load(path_t0.joinpath('ap.npy')).astype(np.float32)
        raw = np.load(path_t0.joinpath('raw.npy')).astype(np.float32)
        spikes = pd.read_parquet(path_t0.joinpath('spikes.pqt')) if path_t0.joinpath('spikes.pqt').exists() else None
        sl = SpikeSortingLoader(pid=pid, one=one)
        channels = sl.load_channels()
        eqcs = {}

        eqcs['ap'] = viewephys(ap[:, int(0.5 * FS): int(0.55 * FS)], fs=FS, title='ap', channels=channels, br=regions)
        eqcs['raw'] = viewephys(raw, fs=FS, title='raw', channels=channels, br=regions)
        for label, eqc in eqcs.items():
            file_png = path_qc.joinpath(f'01_{t0_str}_{label}.png')
            # Plot not good spikes in red
            if spikes is not None:
                sel = np.logical_and(spikes['sample'] > int(0.5 * FS), spikes['sample'] < int(0.55 * FS))

                eqc.ctrl.add_scatter(spikes['sample'][sel] / FS * 1000 - 500, spikes['trace'][sel], (0, 255, 0, 100), label='spikes')
            eqc.ctrl.set_gain(30)
            eqc.viewBox_seismic.setYRange(0, 384)
            eqc.resize(1960, 1200)
            eqc.grab().save(str(file_png))
            eqc.close()
