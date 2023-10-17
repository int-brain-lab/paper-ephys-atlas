'''
Load results of KS test
Plot of KDEs and similarity matrix
'''

import numpy as np

##


def load_ks_result(local_result_path, brain_id='beryl_id', label='2023_W41', test_todo='ks-test', use_debias=False,
                   use_path_only=False):

    if use_path_only:
        local_result_b = local_result_path
    else:  # search according to folder structure
        if use_debias:
            local_result_b = local_result_path.joinpath(brain_id).joinpath('use_debias').joinpath(label)\
                .joinpath(test_todo)
        else:
            local_result_b = local_result_path.joinpath(brain_id).joinpath('normal').joinpath(label)\
                .joinpath(test_todo)

    results_log = np.load(local_result_b.joinpath('results_log.npy'), allow_pickle=True)
    regions = np.load(local_result_b.joinpath('results_regions.npy'), allow_pickle=True)
    features = np.load(local_result_b.joinpath('results_features.npy'), allow_pickle=True)
    # results = np.load(local_result_b.joinpath('results_arr.npy'), allow_pickle=True)
    # results_arr1 = np.load(local_result_b.joinpath('results_arr1.npy'), allow_pickle=True)
    return results_log, regions, features  #, results, results_arr1
