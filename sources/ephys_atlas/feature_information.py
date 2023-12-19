import numpy as np

def feature_overall_entropy(counts, return_quantiles=False):
    '''
    This function computes the overall entropy of a feature, across brain regions.
    1) It first computes the entropy per quantile, using the values of the count per brain region
    in such quantile. For example, if on the first quantile region A has Count=0, and region B has Count=3,
    the entropy for this quantile will be 0.
    2) Then, the overall entropy for the feature is computed as the weighted sum over the quantile entropies.
    :param counts: Dataframe of count per quantile, per region ; for a given feature. [N quantile x N region]
    :return: entropy values for : entropy_feature (single value), vector 1xN quantiles (set return_quantiles to True)
    '''
    # Compute the entropy for each column (i.e. quantile)
    # The probability p is simply the N count divided by the total count for this quantile
    # This is a vector 1xN quantiles
    entropy_quantiles = - np.nansum((p := counts.values / np.nansum(counts.values, axis=0)) * np.log2(p), axis=0)
    # The entropy of the feature overall is the weighed sum of the quantile entropies
    # This is a single value
    entropy_feature = np.sum(entropy_quantiles * np.nansum(counts, axis=0) / np.nansum(counts.values))
    if return_quantiles:
        return entropy_feature, entropy_quantiles
    else:
        return entropy_feature
