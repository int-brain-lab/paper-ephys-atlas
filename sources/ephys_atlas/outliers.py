import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats
import matplotlib.pyplot as plt

def detect_outliers_kde(train_data: np.ndarray, test_data: np.ndarray, kde=None, multithread=False):
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
        kde = kde_fit_params(train_data)
    # Scores are logp
    if not multithread:
        score_train = kde.score_samples(train_data)  # (N,)
        score_test = kde.score_samples(test_data)  # (M,)
    else:
        score_train = parrallel_score_samples(kde, train_data)
        score_test = parrallel_score_samples(kde, test_data)
    # We need to create a matrix
    # Put score train vertically, score test horizontally
    score_train = score_train[:, np.newaxis]
    score_test = score_test[np.newaxis, :]
    # We want the value of the KDE for the train samples to be higher than the test
    out = score_train >= score_test
    # This is the probability for the samples to be outliers
    return out.mean(axis=0)


def kde_fit_params(train_data):
    kde = KernelDensity(bandwidth=np.std(train_data) / 16, kernel='gaussian')
    kde.fit(train_data)
    return kde

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
        kde = kde_fit_params(train_data)
    if not ax:
        fig, ax = plt.subplots()

    score_train = kde.score_samples(train_data)
    ax.scatter(train_data[:,0], np.exp(score_train))
    ax.text(-3.5, 0.31, "Gaussian Kernel Density")

    return ax

import multiprocessing

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    # Taken from
    # https://stackoverflow.com/questions/32607552/scipy-speed-up-kernel-density-estimations-score-sample-method  
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))
