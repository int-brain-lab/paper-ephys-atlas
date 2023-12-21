import pandas as pd
import numpy as np
import scipy.stats
from ephys_atlas.feature_information import feature_overall_entropy, feature_region_entropy

# Create a dict of count per quantile (N=6) / region (N=3)
# This mimics the distribution for a given feature
data = dict()
data['reg_1'] = np.array([0, 1, 2, 1, 0, 0])
data['reg_2'] = np.array([0, 0, 2, 1, 1, 0])
data['reg_3'] = np.array([0, 0, 0, 1, 1, 5])
# Create a dataframe of counts
counts = pd.DataFrame.from_dict(data)
counts = counts.transpose()
nc = np.nansum(counts.values)
assert nc == 15


##
def test_feature_overall_entropy(counts, nc):
    info_gain, entropy_feature, entropy_quantiles, entropy_overall = feature_overall_entropy(counts, return_all=True)
    np.testing.assert_array_almost_equal(entropy_quantiles, [0, 0,  1,  1.5849625,  1, 0],
                                         decimal=6)
    entropy_overall_hand = scipy.stats.entropy(np.array([4, 4, 7]) / nc, base=2)
    np.testing.assert_array_almost_equal(entropy_overall_hand, entropy_overall)


def test_feature_region_entropy(counts, nc):
    # Compute with the function
    info_gain_region, entropy_region, entropy_overall = feature_region_entropy(counts, return_all=True)

    # Compute entropy for a region "by hand"
    entropy_region_hand = list()
    for key in data.keys():
        entropy_region_hand.append(scipy.stats.entropy(data[key] / nc, base=2))
    np.testing.assert_array_almost_equal(entropy_region_hand, entropy_region, decimal=10)

    entropy_overall_hand = scipy.stats.entropy(np.array([0, 1, 4, 3, 2, 5]) / nc, base=2)
    np.testing.assert_array_almost_equal(entropy_overall_hand, entropy_overall)


##
# Run the tests
test_feature_overall_entropy(counts, nc)
test_feature_region_entropy(counts, nc)
