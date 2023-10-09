import matplotlib.pyplot as plt
from iblutil.numerical import ismember
from matplotlib.ticker import MultipleLocator

import numpy as np

from iblatlas.atlas import BrainRegions


def region_bars(atlas_id, feature, label='', regions=None, scale='linear', xlims=None):
    """
    Display one feature for all Beryl regions on histograms with the
    Allen atlas colours ala brainwide map
    :param atlas_id:
    :param feature:
    :param label:
    :param regions:
    :return:
    """
    if regions is None:
        regions = BrainRegions()
    ncols = 3
    fs = 5
    barwidth = 0.6
    nfeats = feature.size

    _, rids, fids = np.intersect1d(regions.id, atlas_id, return_indices=True)
    ordre = np.flipud(np.argsort(rids))
    _feature = feature[fids[ordre]]
    rids = rids[ordre]

    fig, ax = plt.subplots(ncols=ncols, figsize=(ncols * 4 / 3, ncols * 7 / 3))
    xlims = xlims or [np.nanmin(_feature), np.nanmax(_feature)]

    for k, k0 in zip(range(3), reversed(range(3))):
        cind = slice(k0 * nfeats // ncols, (k0 + 1) * nfeats // ncols)
        _colours = regions.rgb[rids[cind], :].astype(np.float32) / 255
        _acronyms = regions.acronym[rids[cind]]
        nbars = _colours.shape[0]

        ax[k].barh(np.arange(nbars), _feature[cind], fill=False, edgecolor=_colours, height=barwidth)
        ax[k].barh(np.arange(nbars), _feature[cind], color=_colours, height=barwidth)
        ax[k].set(xscale=scale, yticks=np.arange(nbars), xlim=xlims)
        ax[k].tick_params(axis='y', pad=19, left=False)
        ax[k].set_yticklabels(_acronyms, fontsize=fs, ha='left')

        # # indicate 10% with black line
        # y_start = np.array([plt.getp(item, 'y') for item in Bars])
        # y_end = y_start + [plt.getp(item, 'height') for item in Bars]
        #
        # ax[k].vlines(0.1 * nclus_nins[:, 0], y_start, y_end,
        #              color='k', linewidth=1)

        for ytick, color in zip(ax[k].get_yticklabels(), _colours):
            ytick.set_color(color)
        ax2 = ax[k].secondary_yaxis("right")
        ax2.tick_params(right=False)
        ax2.set_yticks(np.arange(nbars))
        ax2.set_yticklabels(_feature[cind], fontsize=fs)
        for ytick, color in zip(ax2.get_yticklabels(), _colours):
            ytick.set_color(color)
        ax2.spines['right'].set_visible(False)

        ax[k].spines['bottom'].set_visible(False)
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['left'].set_visible(False)
        ax[k].xaxis.set_ticks_position('top')
        ax[k].xaxis.set_label_position('top')
        if label:
            ax[k].set_xlabel(label)
        ax[k].xaxis.set_minor_locator(MultipleLocator(450))
        plt.setp(ax[k].get_xminorticklabels(), visible=False)

    fig.tight_layout()
    return fig, ax


def plot_cumulative_probas(probas, depths, aids, regions=None, ax=None, legend=False):
    """
    :param probas: (ndepths x nregions) array of probabilities for each region that sum to 1 for each depth
    :param depths: (ndepths) vector of depths
    :param aids: (nregions) vector of atlas_ids
    :param regions: optional: iblatlas.BrainRegion object
    :param ax:
    :param legend:
    :return:
    """
    regions = regions or BrainRegions()
    _, rids = ismember(aids, regions.id)
    cprobas = probas.cumsum(axis=1)
    for i, ir in enumerate(rids):
        ax.fill_betweenx(
            depths,
            cprobas[:, i], label=regions.acronym[ir],
            zorder=-i,
            color=regions.rgb[ir] / 255)
    ax.margins(y=0)
    ax.set_xlim(0, 1)
    ax.set_axisbelow(False)
    if legend:
        ax.legend()
    return ax


def plot_probas_df(probas, regions=None, **kwargs):
    """
    Cumulative probability display of regions predictions
    :param probas:
    :param regions:
    :param ax:
    :param legend:
    :return:
    """
    regions = regions or BrainRegions()
    _, rids = ismember(probas.columns.values, regions.id)
    probas = probas.loc[:, probas.columns[np.argsort(regions.order[regions_indices])]]

    plot_cumulative_probas(probas, depths, aids, regions=None, ax=None, legend=False)
    # need to sort the probability columns as per the Allen order


    # cumsum
    data = probas.values.cumsum(axis=-1)

    for i in np.arange(probas.shape[1]):
        ir = regions.id2index(probas.columns[i])[1][0][0]
        ax.fill_betweenx(
            probas.index.values.astype(np.int16),
            data[:, i], label=regions.acronym[ir],
            zorder=-i,
            color=regions.rgb[ir] / 255)
    ax.margins(y=0)
    ax.set_xlim(0, None)
    ax.set_axisbelow(False)
    if legend:
        ax.legend()
    return ax
