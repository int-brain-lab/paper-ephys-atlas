import matplotlib.pyplot as plt
from iblutil.numerical import ismember
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import compute_summary_stat


def plot_kde(feature, df_voltage, brain_id='cosmos_id', regions_id=None,
             br=None, ax=None, summary=None):
    '''
    Plot KDEs for a given feature
    :param feature: str containing the feature name, e.g. 'peak_to_trough_ratio_log'
    :param df_voltage: dataframe of feature values ; must contain the column brain_id
    :param brain_id: parcelation chosen, e.g. 'cosmos_id' or 'beryl_id'
    :param regions_id: array of region ids to be  plotted (as per brain_id parcelation), e.g. [21, 234]
    :param br: brain region object
    :param ax: axis for plotting
    :param summary: summary statistics
    :return:
    '''
    if regions_id is not None:  # remove all the rows that aren't part of this set of region first
        df_voltage = df_voltage[df_voltage[brain_id].isin(regions_id)]
    if br is None:
        br = BrainRegions()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if summary is None:
        summary = compute_summary_stat(df_voltage, feature)

    caids, cids, _ = np.intersect1d(br.id, np.unique(df_voltage[brain_id]), return_indices=True)
    palette = sns.color_palette(br.rgb[cids] / 255)

    kpl = sns.kdeplot(df_voltage, x=feature, hue=brain_id, fill=False, common_norm=False, palette=palette,
                      legend=True, ax=ax)
    kpl.legend(labels=np.flipud(br.acronym[cids]))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    rec = patches.Rectangle((summary.loc[feature, 'q05'], 0), summary.loc[feature, 'dq'],
                            kpl.get_ylim()[1], alpha=0.1, color='k')
    kpl.add_patch(rec)
    kpl.set(title=feature, xlim=summary.loc[feature, 'median'] + summary.loc[feature, 'dq'] * np.array([-0.5, 0.5]) * 2)
    plt.tight_layout()
    if ax is None:
        return fig, ax


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


'''
Plot utils functions for similarity analysis
'''


def plot_feature_colorbar(features_sort, val_sort, ax=None):
    '''
    Plot a colorbar of the feature importance
    :param features_sort: sorted list of features
    :param val_sort: value of interest (e.g. log(p) value from KS test), sorted
    :param ax:
    :return:
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    plt.imshow(np.expand_dims(val_sort, axis=1))
    ax.set_yticks(np.arange(features_sort.size))
    ax.set_yticklabels(features_sort)
    ax.set_xticks([])
    plt.show()
    if ax is None:
        return fig, ax


def plot_similarity_matrix(mat_plot, regions, ax=None, br=None):
    '''
    Plot the similarity matrix as imshow
    :param mat_plot: matrix to plot
    :param regions: list of regions id as ordered in the matrix
    :param ax:
    :param br:
    :return:
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if br is None:
        br = BrainRegions()
    # Plot
    plt.imshow(mat_plot)
    plt.colorbar()
    plt.show()
    # Set tick labels as brain region acronyms
    regions_ac = br.id2acronym(regions)
    ax.set_xticks(np.arange(regions.size))
    ax.set_xticklabels(regions_ac, rotation=90)
    ax.set_yticks(np.arange(regions.size))
    ax.set_yticklabels(regions_ac)
    if ax is None:
        return fig, ax
