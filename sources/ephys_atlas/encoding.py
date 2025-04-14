"""
This module is aimed at providing convenience functions for decoding and encoding models that map electrophysiological
features to brain regions.
This allows to define stable and reproducible electrophysiological feature sets when experimenting with different models,
and also to split the training and test sets in a reproducible way, stratified by pids
"""

import abc
import pandas as pd
import numpy as np

from iblatlas.atlas import AllenAtlas
from iblutil.numerical import ismember

import ephys_atlas.data

_SEED = 7654


def _train_test_split(df_voltage, test_size=0.25, seed=_SEED, include_benchmarks=True):
    pids = df_voltage.iloc[:, 0].groupby("pid").count()
    if include_benchmarks:
        benchmark_idx, _ = ismember(
            df_voltage.index.get_level_values(0), ephys_atlas.data.BENCHMARK_PIDS
        )
        # this is the number of test channels to add to the benchmark to reach the test_size overall proportion
        ntest_channels = benchmark_idx.size * test_size - np.sum(benchmark_idx)
        pids = pids.iloc[
            ~ismember(pids.index.get_level_values(0), ephys_atlas.data.BENCHMARK_PIDS)[
                0
            ]
        ]
        test_size = ntest_channels / pids.sum()
        fixed_test_pids = ephys_atlas.data.BENCHMARK_PIDS
    else:
        fixed_test_pids = []
    # shuffle pids and take the first n pids that best match the desired proportion test_size
    pids = pids.sample(frac=1, random_state=seed)
    return pids, fixed_test_pids


def train_test_split_folds(df_voltage, N=4, seed=_SEED, include_benchmarks=True):
    """
    if True, will return N folds of training and test sets covering the whole dataset
     in this case N is the closest integer to 1/test_size
    :param df_voltage: voltage features dataframe
    :param N: Number of folds
    :param seed: random seed
    :param include_benchmarks (True): include benchmark probes in the test set
    :return:
    """
    pids, fixed_test_pids = _train_test_split(
        df_voltage, test_size=1 / N, seed=seed, include_benchmarks=include_benchmarks
    )
    folds = np.floor(np.arange(pids.size) / pids.size * N).astype(int)
    index_pids = df_voltage.index.get_level_values(0).values.astype("<U36")
    test_idx_folds = []
    for fold in np.arange(N):
        test_idx, _ = ismember(
            index_pids, np.array(list(pids.index[folds == fold]) + fixed_test_pids)
        )
        test_idx_folds.append(test_idx)
    return test_idx_folds


def train_test_split_indices(
    df_voltage, test_size=0.25, seed=_SEED, include_benchmarks=True
):
    """
    Splits the features dataframe into a training and a test set, stratified by pid, with optionally
    pinning some insertions as test insertions, typically the 13 benchmark insertions scattered accross
    the mouse brain that would always be part of the testing set.
    :param df_voltage: voltage features dataframe
    :param test_size: proportion of the set to be used as test set (0-1)
    :param seed: random seed
    :param include_benchmarks (True): include benchmark probes in the test set
    :return: boolean vectors (nfeats,) train_idx, test_idx
    """
    pids, fixed_test_pids = _train_test_split(
        df_voltage,
        test_size=test_size,
        seed=seed,
        include_benchmarks=include_benchmarks,
    )
    ilast = np.searchsorted(pids.cumsum() / pids.sum(), test_size)
    test_idx, _ = ismember(
        df_voltage.index.get_level_values(0), list(pids.index[:ilast]) + fixed_test_pids
    )
    train_idx = ~test_idx
    return train_idx, test_idx


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, X, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abc.abstractmethod
    def score(self, X, y, **kwargs):
        pass

    @staticmethod
    def split_dataframe(df, seed=_SEED, frac=0.1):
        """
        Splits a dataframe in a training and a validation dataframe.
        :param df:
        :param seed:
        :param frac: fraction of the dataframe kept for tests
        :return:
        """
        nrecs = df.shape[0]
        xmask = np.ones(nrecs, dtype=np.bool_)
        np.random.seed(seed)
        xmask[np.random.randint(0, nrecs, size=int(np.floor(nrecs * frac)))] = False
        df_training = df.iloc[xmask, :].copy()
        df_test = df.iloc[~xmask, :].copy()
        return df_training, df_test


class NullModel01(AbstractModel):
    """
    Suggested null model #1: Predicted firing rate is the average firing rate of all training dataset
    channels that were located within the same Beryl region as the test channel
    """

    def __init__(self, ba=None):
        self.df_regions = None  # see the fit function
        self.atlas = AllenAtlas() if ba is None else ba

    def predict(self, atlas_id):
        isin, idfr = ismember(atlas_id, self.df_regions.index)
        return self.df_regions.iloc[idfr, :], isin

    def predict_xyz(self, X):
        """
        :param X: np.array of x, y, z coordinates or dataframe containing x, y and z fields
        :param kwargs:
        :return:
        """
        # fixme: this is outdated
        if self.df_regions is None:
            return
        # the input features X is a 3 columns array of xyz coordinates
        if isinstance(X, pd.DataFrame):
            X = X[["x", "y", "z"]].values
        aids = self.atlas.get_labels(X, mapping="Beryl")
        # here the prediction is a simple lookup of the existing dataframe
        isin, idfr = ismember(aids, self.df_regions.index)
        return self.df_regions.iloc[idfr]["spike_rate"]["median"].values, isin

    def fit(self, df_voltage, x_list, y_name="atlas_id"):
        """

        :param df_voltage:
        :param xlist:
        :param yname: 'atlas_id' or 'beryl_id' or 'cosmos_id'
        :return:
        """
        # percentiles = np.arange(10) / 10
        # quantile_funcs = [(p, lambda x: x.quantile(p)) for p in percentiles]
        self.df_regions = df_voltage.groupby(y_name).agg(
            **{x: pd.NamedAgg(column=x, aggfunc="median") for x in x_list},
            channel_count=pd.NamedAgg(column=x_list[0], aggfunc="count"),
        )

    def score(self, X, y, **kwargs):
        pass
