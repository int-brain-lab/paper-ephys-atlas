import matplotlib.pyplot as plt
from iblutil.numerical import ismember
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
from iblatlas.atlas import BrainRegions
from ephys_atlas.data import compute_summary_stat
from ephys_atlas.encoding import FEATURES_LIST
from matplotlib import cm  # This is deprecated, but cannot import matplotlib.colormaps as cm
from brainbox.plot_base import ProbePlot, arrange_channels2banks, plot_probe
from brainbox.ephys_plots import plot_brain_regions


def color_map_feature(feature_list=FEATURES_LIST, cmap='Pastel1_r', n_inc=12):
    # color_map = cm.get_cmap(cmap, n_inc)
    # np.linspace(0, 1, num=len(feature_list))
    # color_alpha = color_map(np.linspace(0, 1, num=len(feature_list)))
    # color_only = color_alpha[:, 0:3]  # Return only the color values
    # # Convert to list of tuple [(0, 0.2, 0.1), (...)]
    # list_col = color_only.tolist()
    # list_out = list()
    # for i_col in list_col:
    #     list_out.append(tuple(i_col))
    # TODO above is correct but umpractical ?
    list_out = ['m', 'g', 'm', 'b']
    assert len(list_out) == len(FEATURES_LIST)

    return list_out


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
        # df_voltage = df_voltage.loc[df_voltage[brain_id].isin(regions_id)]
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


def prepare_data_probe_plot(data_arr, xy, cmap=None, clim=None):
    '''
    Prepare the data to use in probe_plot
    Example usage:

    from brainbox.plot_base import plot_probe
    data = prepare_data_probe_plot(your_feature_data, xy)
    plot_probe(data.convert2dict())

    :param data_arr: Vector of data (feature) to plot for each channel [N channel x 1]
    :param xy: Matrix of spatial channel position (in um), lateral_um (x) and axial_um (y) [N channel x2]
    :param cmap: color map
    :param clim: color bar limit ; by default it uses the quantiles of the distribution across channels
    :return:
    '''
    data_bank, x_bank, y_bank = arrange_channels2banks(data_arr, xy)
    data = ProbePlot(data_bank, x=x_bank, y=y_bank, cmap=cmap)

    if clim is None:
        clim = np.nanquantile(np.concatenate([np.squeeze(np.ravel(d)) for d in data_bank]).ravel(),
                              [0.1, 0.9])
    data.set_clim(clim)
    return data


def figure_features_chspace(pid_df, features, xy):
    '''

    :param pid_df: Dataframe containing channels and voltage information for a given PID
    Example on how to prepare it:
    # Merge the voltage and channels dataframe
    df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True).dropna()
    # Select a PID and create the single probe dataframe
    pid = '0228bcfd-632e-49bd-acd4-c334cf9213e9'
    pid_df = df_voltage[df_voltage.index.get_level_values(0).isin([pid])].copy()

    :param features: list of feature names to display, e.g. ['rms_lf', 'psd_delta', 'rms_ap']
    These have to bey columns keys of the pid_df
    :param xy: Matrix of spatial channel position (in um), lateral_um (x) and axial_um (y) [N channel x2]
    :return:
    '''
    fig, axs = plt.subplots(1, len(features) + 2, sharey=True)

    for i_feat, feature in enumerate(features):
        feat_arr = pid_df[[feature]].to_numpy()
        # Plot feature
        data = prepare_data_probe_plot(feat_arr, xy)
        plot_probe(data.convert2dict(), ax=axs[i_feat], show_cbar=False)
        del data
        axs[i_feat].set_title(feature)

    # Plot brain region in space in unique colors
    d_uni = np.unique(pid_df['atlas_id'].to_numpy(), return_inverse=True)[1]
    d_uni = d_uni.astype(np.float32)
    data = prepare_data_probe_plot(d_uni, xy)
    plot_probe(data.convert2dict(), ax=axs[len(features)], show_cbar=False)
    axs[len(features)].set_title('atlas reg')
    # TODO color code based on brain region color
    # region_info = br.get(pid_ch_df['atlas_id'])
    # region_info.rgb

    # Plot brain region along probe depth with color code
    plot_brain_regions(pid_df['atlas_id'], channel_depths=pid_df['axial_um'].to_numpy(),
                       ax=axs[len(features) + 1])
    axs[len(features) + 1].set_title('brain region')
