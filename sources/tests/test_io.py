from pathlib import Path
import unittest

import pandas as pd
import numpy as np

import ephys_atlas.data
from neuropixel import trace_header
# TODO create test fixtures with a few waveforms right now this work only on my laptop (olivier)

th = trace_header(version=1)
xy = th["x"] + 1j * th["y"]


class TestWaveformsCoordinates(unittest.TestCase):
    def test_get_channels_distances(self):
        mat = ephys_atlas.data._get_channel_distances_indices(xy, extract_radius_um=200)
        assert mat.shape == (384, 40)

    def test_get_waveform_coordinates(self):
        EPHYS_SAMPLES_PATH = Path("/datadisk/Data/paper-ephys-atlas/ephys-atlas-sample")
        EXTRACT_RADIUS = 200
        pid = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
        sample = "T00500"
        sample_folder = EPHYS_SAMPLES_PATH.joinpath(pid, sample)
        file_spikes = sample_folder.joinpath("spikes.pqt")
        file_waveforms = sample_folder.joinpath("waveforms.npy")
        df_spikes = pd.read_parquet(file_spikes)
        waveforms = np.load(file_waveforms)
        wxy = ephys_atlas.data.get_waveforms_coordinates(
            df_spikes["trace"], return_complex=True
        )

        # test a random waveform
        iw = 182
        cind = np.abs(xy[int(df_spikes["trace"].iloc[iw])] - xy) <= EXTRACT_RADIUS
        assert np.all(xy[cind] == wxy[iw])

        # test that the NaNs pattern of the waveforms check out with the NaNs pattern of the reconstructed coordinates
        for iw in np.arange(waveforms.shape[0]):
            np.testing.assert_array_equal(
                np.all(np.isnan(waveforms[iw, :, :]), axis=0), np.isnan(wxy[iw])
            )
