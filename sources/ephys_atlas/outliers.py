import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats
import matplotlib.pyplot as plt

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


def detect_outlier_kstest(train_data: np.ndarray, test_data: np.ndarray):
    '''
    For a single feature, compute channel by channel the KS test against the distribution

    Parameters:
    - train_data: (N,) numpy array, training dataset (assumed to represent the true distribution).
    - test_data: (M,) numpy array, test dataset (points to evaluate for outlier probability).

    Returns:
    - outlier_statistic: (M,) numpy array, KS statistic of each test sample being an outlier.
    '''
    out = np.zeros(test_data.shape)
    for count, sample in enumerate(test_data):  # Test on each channel value independently
        ks_stat = stats.kstest(sample, train_data)
        out[count] = ks_stat.statistic
    return out


def plot_kde_fit(train_data: np.ndarray, ax = None, kde = None):
    if not kde:
        kde = KernelDensity()
        kde.fit(train_data)
    if not ax:
        fig, ax = plt.subplots()

    score_train = kde.score_samples(train_data)
    ax.scatter(train_data[:,0], np.exp(score_train))
    ax.text(-3.5, 0.31, "Gaussian Kernel Density")

    return ax