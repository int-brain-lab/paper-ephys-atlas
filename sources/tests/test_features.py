from pathlib import Path
import unittest

import numpy as np

import neuropixel
import ephys_atlas.features

# np.save("/home/olivier/scratch/lfp_destriped.npy", des_lf.astype(np.float16))
# np.save("/home/olivier/scratch/ap_destriped.npy", des_ap.astype(np.float16))


TEST_DATA_PATH = Path("/home/olivier/scratch")


class TestFeatureSets(unittest.TestCase):

    def test_sets(self):
        assert len(ephys_atlas.features.voltage_features_set('all')) == 32
        assert len(ephys_atlas.features.voltage_features_set('raw_ap')) == 2
        assert len(ephys_atlas.features.voltage_features_set()) == 23


class TestLFPFeatures(unittest.TestCase):
    def setUp(self):
        self.data_lf = np.load(TEST_DATA_PATH / "lfp_destriped.npy").astype(np.float32)

    def test_csd(self):
        df = ephys_atlas.features.csd(
            self.data_lf, fs=2500, geometry=neuropixel.trace_header(version=1)
        )
        self.assertTrue(df.shape[0] == self.data_lf.shape[0])

    def test_lf(self):
        df = ephys_atlas.features.lf(self.data_lf, fs=2500)
        self.assertTrue(df.shape[0] == self.data_lf.shape[0])


class TestAPFeatures(unittest.TestCase):
    def setUp(self):
        self.data_ap = np.load(TEST_DATA_PATH / "ap_destriped.npy").astype(np.float32)

    def test_ap(self):
        df = ephys_atlas.features.ap(
            self.data_ap[:, 10_000:11_000], geometry=neuropixel.trace_header(version=1)
        )
        self.assertTrue(df.shape[0] == self.data_ap.shape[0])


class TestWaveformFeatures(unittest.TestCase):
    def setUp(self):
        self.data_ap = np.load(TEST_DATA_PATH / "ap_destriped.npy").astype(np.float32)

    def test_ap(self):
        df, waveforms = ephys_atlas.features.spikes(
            self.data_ap[:, 10_000:11_000],
            fs=30_000,
            geometry=neuropixel.trace_header(version=1),
            return_waveforms=True,
        )
        self.assertTrue(df.shape[0] == waveforms["df_spikes"]["channel"].nunique())
        self.assertEqual(4, len(waveforms.keys()))
