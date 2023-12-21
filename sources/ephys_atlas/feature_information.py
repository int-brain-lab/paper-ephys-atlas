import numpy as np
import scipy.stats

def feature_overall_entropy(counts, return_all=False):
    '''
    This function computes the overall entropy of a feature, across brain regions.
    1) It first computes the entropy per quantile, using the values of the count per brain region
    in such quantile. For example, if on the first quantile region A has Count=0, and region B has Count=3,
    the entropy for this quantile will be 0.
    2) Then, the overall entropy for the feature is computed as the weighted sum over the quantile entropies.
    :param counts: Dataframe of count per quantile, per region ; for a given feature. [N quantile x N region]
    :param return_all: return the underlying entropies if set to True
    :return:
    - info_gain (single value) : the information gain (bits) [DEFAULT]
    - entropy_feature (single value): the entropy of the feature overall is the weighed sum of the quantile entropies
    - entropy_quantiles (vector 1xN quantiles): entropy for each quantile across brain regions
    - entropy_overall (single value): the entropy overall is the entropy over the sum of counts per brain region
    '''
    # Count the overall number of values (== number of channels)
    nc = np.nansum(counts.values)
    # Compute the entropy for each column (i.e. quantile)
    # The probability p is simply the N count divided by the total count for this quantile
    # This is a vector 1xN quantiles
    entropy_quantiles = - np.nansum((p := counts.values / np.nansum(counts.values, axis=0)) * np.log2(p), axis=0)
    # The entropy of the feature overall is the weighed sum of the quantile entropies
    # This is a single value
    entropy_feature = np.sum(entropy_quantiles * np.nansum(counts, axis=0) / nc)
    # Compute entropy overall for this feature, by summing the counts over each brain region (single value)
    entropy_overall = scipy.stats.entropy(np.nansum(counts.values, axis=1) / nc, base=2)
    # Compute information gain (single value, for this feature)
    info_gain = entropy_overall - entropy_feature
    if return_all:
        return info_gain, entropy_feature, entropy_quantiles, entropy_overall
    else:
        return info_gain


def feature_region_entropy(counts, return_all=False, normalise=False):
    '''
    This function computes the entropy of a feature, for a given brain region.
    It computes the entropy across the quantiles of a brain region.
    :param counts: Dataframe of count per quantile, per region ; for a given feature. [N quantile x N region]
    :return:
    - info_gain_region (vector 1xN region) : the information gain per region (bits) [DEFAULT]
    - entropy_region (vector 1xN region): the entropy of the feature for a region, across the quantiles
    - entropy_overall (single value): the entropy overall is computed by summing across brain regions
    the values in each quantile, then computing the entropy across quantiles
    '''
    # Count the overall number of values (== number of channels) ; this is a single value
    nc = np.nansum(counts.values)
    # Transpose (n_quantiles, n_regions)
    counts = counts.transpose()
    nc_reg = np.nansum(counts.values, axis=0)  # Number of channels per region (vector)
    # Compute the entropy for each column i.e. region
    # TODO dividing by the N channels of the region biases the information gain to be high for regions with low N chan
    # TODO Use nc instead ?
    entropy_region = - np.nansum((f := counts.values / nc_reg) * np.log2(f), axis=0)
    # Compute the entropy overall for this feature by summing the values in each quantile
    # (this is a single value)
    entropy_overall = scipy.stats.entropy(np.nansum(counts, axis=1) / nc, base=2)
    # Compute information gain per region (vector 1xN region)
    # TODO TESTING Weighted by number of channels
    if normalise:
        info_gain_region = (entropy_overall * nc - entropy_region * nc_reg) / (nc + nc_reg)
    else:
        info_gain_region = entropy_overall - entropy_region
    if return_all:
        return info_gain_region, entropy_region, entropy_overall
    else:
        return info_gain_region


