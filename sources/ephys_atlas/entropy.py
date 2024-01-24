import pandas as pd
import numpy as np
from ephys_atlas.feature_information import feature_overall_entropy


def compute_info_gain(df_voltage, feature, mapping,
                      save_folder=None, save_name='info_gain'):
    '''
    Compute the information gain for each pair of region, for one given feature
    :param df_voltage: the dataframe of features
    :param feature: string, e.g. 'rms_ap'
    :param mapping: string, e.g. 'Cosmos'
    :param save_folder: path to save folder
    :return: dataframe of (Nregion * Nregion) rows x 4 columns (reg1, reg2, info_gain, feature)
    '''
    quantiles = df_voltage[feature].quantile(np.linspace(0, 1, 600)[1:])
    quantiles = np.searchsorted(quantiles, df_voltage[feature])
    # Create table of shape (n_regions, n_quantiles) that contains the count
    counts = pd.pivot_table(df_voltage, values=feature, index=mapping + '_id', columns=quantiles, aggfunc='count')

    # Create a dataframe of Nregion X Nregion that will contain the entropy computed for a pair of region
    df_entropy = pd.DataFrame(index=counts.index.rename(mapping + '_id_reg1'),
                              columns=counts.index.rename(mapping + '_id_reg2'))

    # Divide the counts into 2 regions
    for ireg1, reg1 in enumerate(counts.index):
        for reg2 in counts.index[ireg1 + 1:]:
            counts_reg = counts.loc[[reg1, reg2], :]
            information_gain = feature_overall_entropy(counts_reg)

            # Save the result in both place in the DF
            df_entropy.at[reg1, reg2] = information_gain
            df_entropy.at[reg2, reg1] = information_gain

    # Unstack (flatten the DF) and add feature
    # The dataframe will become of size (Nregion * Nregion) x 4
    df_entropy = df_entropy.unstack().reset_index()
    df_entropy.rename(columns={0: 'info_gain'}, inplace=True)
    df_entropy['feature'] = feature

    if save_folder is not None:  # Save dataframe
        df_entropy.to_parquet(save_folder.joinpath(f'{feature}__{mapping}__info_gain.pqt'))

    return df_entropy
