import numpy as np
from sklearn.neighbors import KernelDensity


def detect_outliers_kde(train_data: np.ndarray, test_data: np.ndarray, kde=None):
    """
    Detects outliers in D-dimensional space using Kernel Density Estimation (KDE).

    Parameters:
    - train_data: (N, D) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M, D) numpy array, test dataset (points to evaluate for outlier probability).

    Use in the Ephys Atlas analysis:
    D are the numbers of features. N and M are the number of channel for train and test sets.

    Returns:
    - outlier_probs: (M,) numpy array, probability of each test sample being an outlier.
    """
    # If kde is set, it is assumed to be alread trained
    if not kde:
        kde = KernelDensity()
        kde.fit(train_data)
    # Scores are logp
    score_train = kde.score_samples(train_data)  # (N,)
    score_test = kde.score_samples(test_data)  # (M,)
    # We need to create a matrix
    # Put score train vertically, score test horizontally
    score_train = score_train[:, np.newaxis]
    score_test = score_test[np.newaxis, :]
    # We want the value of the KDE for the train samples to be higher than the test
    out = score_train >= score_test
    # This is the probability for the samples to be outliers
    return out.mean(axis=0)
