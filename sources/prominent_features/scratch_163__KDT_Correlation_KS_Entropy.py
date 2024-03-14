import numpy as np
from pathlib import Path
import pandas as pd
from iblatlas.atlas import BrainRegions
from scipy import stats
from one.api import ONE

one = ONE()
br = BrainRegions()

label = '2023_W51'
mapping = 'Allen'

local_data_path_ks = Path(f'/Users/gaelle/Documents/Work/EphysAtlas/Entropy_KS/{label}/')
local_data_path_en = Path(f'/Users/gaelle/Documents/Work/EphysAtlas/Entropy_DF_WF/{label}/')
save_folder = Path(f'/Users/gaelle/Documents/Work/EphysAtlas/ks_en_comp/{label}/')

# local_data_path_ks = Path(f'/mnt/s0/ephys-atlas-decoding/kstest/{label}/')
# local_data_path_en = Path(f'/mnt/s0/ephys-atlas-decoding/entropy/{label}/')
# save_folder = Path(f'/mnt/s0/ephys-atlas-decoding/ks_en_comp/{label}/')

if not save_folder.parent.exists():
    save_folder.parent.mkdir()
if not save_folder.exists():
    save_folder.mkdir()

# Load dataframes for KS + entropy
df_info_ks = pd.read_parquet(local_data_path_ks.joinpath(f'{label}__{mapping}__overall_ks_test.pqt'))
df_info_en = pd.read_parquet(local_data_path_en.joinpath(f'{label}__{mapping}__overall_info_gain.pqt'))

features = df_info_ks['feature'].unique()

##
# https://gist.github.com/DominikPeters/211aa0eb89599189c6d18286142ec15e


def cayley_distance(x, y):
    A = range(len(x))
    inv_y = tuple(y.index(a) for a in A)
    comp = tuple(x[inv_y[a]] for a in A)
    cycles = 0
    rem = set(A)
    while rem:
        a = rem.pop()
        cycles += 1
        while comp[a] in rem:
            a = comp[a]
            rem.remove(a)
    return len(A) - cycles

##
# Create pivot table of brain region comparison
# Columns are features, multi-indices are brain regions compared (reg1 / reg 2)

# === KS === Note: We use only KS statistic, not p-value for this analysis
df_multi_ks = pd.pivot_table(df_info_ks, values='statistic',
                             index=[f'{mapping}_id_reg1', f'{mapping}_id_reg2'],
                             columns=['feature'])
# Replace Nans with zeros
df_multi_ks.fillna(0, inplace=True)

# == Entropy ===
df_multi_en = pd.pivot_table(df_info_en, values='info_gain',
                             index=[f'{mapping}_id_reg1', f'{mapping}_id_reg2'],
                             columns=['feature'])
# Replace Nans with zeros
df_multi_en.fillna(0, inplace=True)

##
# Get "information gain" by averaging the statistic (entropy or KS distance) over regions
# This gives a dataframe of N regions x N features
# The highest the value is, the most informative it is (WARNING: only do such ranking per brain region with the entropy)
information_gain_ks = df_multi_ks.groupby(level=[0]).mean()
information_gain_en = df_multi_en.groupby(level=[0]).mean()
# Save
information_gain_ks.to_parquet(save_folder.joinpath(f'{label}__{mapping}__information_gain_ks.pqt'))
information_gain_en.to_parquet(save_folder.joinpath(f'{label}__{mapping}__information_gain_en.pqt'))

##
# Compute correlation between ranking, for each brain region

df_kdt = pd.DataFrame(index=information_gain_ks.index.rename(mapping + '_id'),
                      columns=['kdt_correlation', 'kdt_pvalue', 'kdt_statistic',
                               'cayley_dist'])

df_kdt = df_kdt.reset_index()

for br_id in information_gain_ks.index:

    df_serie_ks = information_gain_ks.loc[br_id].to_frame('ks_statistic')
    df_serie_en = information_gain_en.loc[br_id].to_frame('entropy')

    df_merge = df_serie_ks.join(df_serie_en)

    df_merge['entropy_argsort'] = np.argsort(df_merge['entropy'].to_numpy())
    df_merge['ks_statistic_argsort'] = np.argsort(df_merge['ks_statistic'].to_numpy())

    # Compute correlation
    correlation = stats.kendalltau(df_merge['entropy_argsort'].to_list(),
                                   df_merge['ks_statistic_argsort'].to_list())

    cayley_dist = cayley_distance(df_merge['entropy_argsort'].to_list(),
                                  df_merge['ks_statistic_argsort'].to_list())
    # Save
    df_kdt.loc[(df_kdt[mapping + '_id'] == br_id), 'kdt_correlation'] = correlation.correlation
    df_kdt.loc[(df_kdt[mapping + '_id'] == br_id), 'kdt_pvalue'] = correlation.pvalue
    df_kdt.loc[(df_kdt[mapping + '_id'] == br_id), 'kdt_statistic'] = correlation.statistic
    df_kdt.loc[(df_kdt[mapping + '_id'] == br_id), 'cayley_dist'] = cayley_dist

df_kdt.to_parquet(save_folder.joinpath(f'{label}__{mapping}__kdt_ks_en.pqt'))

##
br_id = 591  # 44
df_ks = information_gain_ks.loc[br_id].to_frame('value')
print(df_ks.sort_values(by=['value'], ascending=False))

df_en = information_gain_en.loc[br_id].to_frame('value')
print(df_en.sort_values(by=['value'], ascending=False))

##
# Print distinctive features between 2 regions
df_2reg = df_multi_ks.loc[br.acronym2id('PO')[0], br.acronym2id('LP')[0]]
df_2reg.sort_values(ascending=False)

'''
For PO and LP, top features are according to KS-distance:

psd_theta               0.489766  -> pvalue = 0
psd_alpha               0.489078  -> pvalue = 0
rms_lf                  0.442379
psd_beta                0.424650
psd_gamma               0.384942
psd_delta               0.371318
rms_ap                  0.283109
spike_count             0.144857
trough_val              0.098181
peak_val                0.097786
tip_time_secs           0.091370
polarity                0.090827
repolarisation_slope    0.074162
recovery_time_secs      0.072529
trough_time_secs        0.072529
recovery_slope          0.065508
alpha_std               0.060223
peak_time_secs          0.051394
depolarisation_slope    0.042968
tip_val                 0.042132
alpha_mean              0.023185
'''
##
# Print max feature
inf_display = information_gain_ks.copy()

inf_display['max_feat'] = inf_display.idxmax(axis=1)
print(inf_display.groupby("max_feat").count().sort_values(by='alpha_mean'))

# inf_display.loc[inf_display['max_feat'] == 'polarity']
br.id2acronym(inf_display.loc[inf_display['max_feat'] == 'polarity'].index)
