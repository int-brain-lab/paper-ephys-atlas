import unittest
import ephys_atlas.plots

class TestPlots(unittest.TestCase):

    def test_colormaps(self):
        ephys_atlas.plots.color_map_feature()
