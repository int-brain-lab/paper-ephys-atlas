'''
Code to compute the KS-test, per brain region pair 1-to-1
'''

from pathlib import Path
from ephys_atlas.data import prepare_df_voltage
from scipy.stats import ks_2samp
from scipy.special import kl_div, rel_entr
import pandas as pd
import numpy as np
import ephys_atlas.data
import ephys_atlas.plots
from ephys_atlas.encoding import voltage_features_set
from iblatlas.atlas import BrainRegions
from one.api import ONE
import os

one = ONE(mode='remote')
br = BrainRegions()

local_path = Path("/Users/gaelle/Documents/Work/EphysAtlas/Data/")
local_result = local_path.parent.joinpath('Fig3_Result')
label = '2023_W34'
USE_DEBIAS = False

# Select brain region level to do analysis over
brain_id = 'cosmos_id'

# Select test to perform
test_todo = 'ks-test'  # ks-test ; kl-test

# Select set of features to do analysis over
features = ['alpha_mean', 'alpha_std', 'spike_count', 'peak_time_secs', 'peak_val',
            'trough_time_secs', 'trough_val', 'tip_time_secs', 'tip_val',
            'recovery_time_secs', 'depolarisation_slope', 'repolarisation_slope',
            'recovery_slope', 'polarity', 'rms_ap', 'rms_lf', 'psd_delta',
            'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma', 'psd_lfp',
            'rms_lf_csd', 'psd_delta_csd', 'psd_theta_csd', 'psd_alpha_csd',
            'psd_beta_csd', 'psd_gamma_csd', 'spike_count_log',
            'peak_to_trough_duration', 'peak_to_trough_ratio', 'peak_to_trough_ratio_log']

# TODO the function prepare_df_voltage adds in features
FEATURE_SET = ['raw_ap', 'raw_lf', 'localisation', 'waveforms']
x_list = voltage_features_set(FEATURE_SET)

##
# Load and prepare dataframe
if USE_DEBIAS:
    _, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(
        local_path, label=label, one=one, verify=True)
    df_voltage_debias = pd.read_parquet(local_path.joinpath('df_voltage_debias.pqt'))
    df_voltage = df_voltage_debias
else:
    df_voltage, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(
        local_path, label=label, one=one, verify=True)

df_voltage = prepare_df_voltage(df_voltage, df_channels)

##
# Regions of DF
regions = df_voltage[brain_id].unique()
# reorganise order
if brain_id == 'cosmos_id':
    regions = br.acronym2id(['Isocortex', 'HPF', 'CTXsp', 'OLF', 'CNU', 'HY', 'TH', 'MB', 'HB', 'CB'])


##
# Run test 1-by-1

# Create an array containing two regions to compare
reg_arr = np.empty(shape=[regions.size, regions.size, 2])  # 10x10x2 for Cosmos
for i_reg, reg in enumerate(regions):  # Fill in the array
    reg_arr[i_reg, :, 0] = reg
    reg_arr[:, i_reg, 1] = reg  # cf TODO Note: this is not used currently but could be useful if re-written

# Fill in result array with Nans
results_arr = np.empty(shape=[regions.size, regions.size, len(features)])
results_arr[:] = np.nan
results_arr1 = results_arr.copy()

# Run over the half portion of the array
for i_reg, reg in enumerate(regions):
    reg_comp_arr = reg_arr[i_reg+1:, i_reg, 0]  # Regions to compare against start from the next available
    print(f'-- Region: {reg}, {i_reg}/{len(regions)} ----')
    # Prepare dataframe per brain regions
    brain_groupby = df_voltage.groupby(brain_id)
    df_reg = brain_groupby.get_group(reg)

    for i_reg_comp, reg_comp in enumerate(reg_comp_arr):
        df_reg_comp = brain_groupby.get_group(reg_comp)
        # print(f'-- Region compare: {reg_comp}, {i_reg+i_reg_comp+1} ----')

        for i_feat, feat in enumerate(features):
            # Distributions to compare
            x = df_reg[feat]
            y = df_reg_comp[feat]
            # Increments in the similarity matrix for storage
            inc0 = i_reg
            inc1 = i_reg+i_reg_comp+1
            inc2 = i_feat
            if test_todo == 'ks-test':
                ks = ks_2samp(x, y)
                results_arr[inc0, inc1, inc2] = ks.pvalue
                results_arr1[inc0, inc1, inc2] = ks.statistic
            elif test_todo == 'kl-test':
                # prepare probability distribution function (sum of probabilities = 1) ; create histograms
                bin_size = 200
                # Use same range, make sure bins are the same across the 2 PDFs
                range = (np.min(np.hstack((x, y))), np.max(np.hstack((x, y))))
                x_bin, x_edge = np.histogram(x, bins=bin_size, range=range)
                y_bin, y_edge = np.histogram(y, bins=bin_size, range=range)
                np.testing.assert_equal(x_edge, y_edge)
                # Divide by N samples to get PDF
                x_pdf = x_bin / len(x)
                y_pdf = y_bin / len(y)
                # Measure divergence and sum (these are non-symmetrical)
                kl = kl_div(x_pdf, y_pdf)
                kl_r = rel_entr(x_pdf, y_pdf)
                results_arr[inc0, inc1, inc2] = sum(kl)
                results_arr1[inc0, inc1, inc2] = sum(kl_r)

results_log = np.log10(results_arr)

# TODO Note the code above could be streamlined for readability using indices, but should be as fast:
'''
# Get indices over which to do the comparison
indices = np.triu_indices_from(np.squeeze(reg_arr[:, :, 0]))
# Need to remove diagonal terms
diagonal = np.diag_indices(regions.size)
'''

##
# Save
if USE_DEBIAS:
    local_result_b = local_result.joinpath(brain_id).joinpath('use_debias').joinpath(label).joinpath(test_todo)
else:
    local_result_b = local_result.joinpath(brain_id).joinpath('normal').joinpath(label).joinpath(test_todo)

if not local_result_b.exists():
    os.makedirs(local_result_b)

np.save(local_result_b.joinpath('results_arr.npy'), results_arr, allow_pickle=True)
np.save(local_result_b.joinpath('results_arr1.npy'), results_arr1, allow_pickle=True)
np.save(local_result_b.joinpath('results_log.npy'), results_log, allow_pickle=True)
np.save(local_result_b.joinpath('results_regions.npy'), regions, allow_pickle=True)
np.save(local_result_b.joinpath('results_features.npy'), features, allow_pickle=True)
