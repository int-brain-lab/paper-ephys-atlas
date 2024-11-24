import numpy as np


def compute_ch_eudistance(n_ch, xyz_arr):
    # Compute the euclidian distance across channels
    # Brute force it (DUPLICATE, NOT SMART, TO GET GOING)
    dist_arr = np.zeros([n_ch, n_ch])
    for ch_1 in range(0, n_ch):
        vect_1 = xyz_arr[ch_1, :]
        for ch_2 in range(0, n_ch):
            vect_2 = xyz_arr[ch_2, :]
            # Compute eucl dist
            dist_arr[ch_1, ch_2] = np.linalg.norm(vect_1 - vect_2)
    return dist_arr


def compute_feature_diff(n_ch, feat_arr):
    # Compute the feature difference across channels
    # Brute force it (DUPLICATE, NOT SMART, TO GET GOING)
    featdiff_arr = np.zeros([n_ch, n_ch])
    for ch_1 in range(0, n_ch):
        val_1 = feat_arr[ch_1]
        for ch_2 in range(0, n_ch):
            val_2 = feat_arr[ch_2]
            # Compute feature diff
            featdiff_arr[ch_1, ch_2] = val_1 - val_2
    return featdiff_arr


def gaussian_weights(x, mu=0, sig=200, d_min=10):
    # Create a truncated gaussian filter based on distance
    # x is the distance between channels
    gauss_f = (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )
    gauss_f[np.where(x < d_min)] = 0
    return gauss_f


def compute_spatial_diff(n_ch, dist_arr, featdiff_arr):
    # TODO bring back argument for gaussian weight as input
    # weighted distance
    weigh_arr = np.zeros([n_ch, n_ch])
    for ch_1 in range(0, n_ch):
        weigh_arr[ch_1, :] = gaussian_weights(dist_arr[ch_1, :])

    # Compute dot product between feature and distance
    dot_arr = np.multiply(featdiff_arr, weigh_arr)
    # Take absolute value
    abs_arr = np.abs(dot_arr)
    # sum over channels (rows) to get one value per channel
    sum_ch = np.sum(abs_arr, axis=1)
    return sum_ch


def meta_spatial_derivative(pid_df, feature):
    """
    Compute spatial derivative
    :param pid_df: Dataframe containing channels and voltage information for a given PID
    :param feature: single string of feature name, e.g. 'rms_lf'
    It has to be a column key of pid_df
    :return:
    """
    # Note: our probes are flat (2D) but it would be easy to add
    # pid_ch_df['z_um'] = 0
    # and xyz_arr = pid_df[['lateral_um', 'axial_um', 'z_um']].to_numpy()
    # Not done here to maximise speed of compute

    # Get N channels
    n_ch = len(pid_df)

    # Get specific feature values as array
    feat_arr = pid_df[[feature]].to_numpy()

    # Create numpy array of xyz um
    xyz_arr = pid_df[["lateral_um", "axial_um"]].to_numpy()

    # Compute the eu distance, the feature difference and the weighted sum across channels
    dist_arr = compute_ch_eudistance(n_ch, xyz_arr)
    featdiff_arr = compute_feature_diff(n_ch, feat_arr)
    sum_ch = compute_spatial_diff(n_ch, dist_arr, featdiff_arr)
    return sum_ch
