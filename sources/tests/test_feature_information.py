import pandas as pd
import numpy as np
from ephys_atlas.feature_information import feature_overall_entropy

# Create a dict of count per quantile (N=6) / region (N=3)
# This mimics the distribution for a given feature
data = {}
data['reg_1'] = np.array([0, 1, 2, 1, 0, 0])
data['reg_2'] = np.array([0, 0, 2, 1, 1, 0])
data['reg_3'] = np.array([0, 0, 0, 1, 1, 5])
# Create a dataframe of counts
counts = pd.DataFrame.from_dict(data)
counts = counts.transpose()

def test_feature_overall_entropy(counts):
    e_feat, e_quant_vect = feature_overall_entropy(counts, return_quantiles=True)
    np.testing.assert_array_almost_equal(e_quant_vect, [0, 0,  1,  1.5849625,  1, 0],
                                         decimal=6)
