'''
brain region uncertainty
'''

import numpy as np


def func_ax_indx(ax_coord, point, radius, volume):
    # ax_coord = 0 : x ;   1: y ;   2: z

    x1 = point[ax_coord] + radius[ax_coord][0]
    x2 = point[ax_coord] + radius[ax_coord][1]

    if x1<0:
        x1=0
    if x2>volume.shape[ax_coord]:
        x2=volume.shape[ax_coord]-1

    return x1, x2


def get_small_volume(point, volume, radius):
    '''
    :param point: 1x3 vector (x,z,y) index of point of interest
    :param volume: nxmxp overall matrix containing labels (int+)
    :param radius: 1x3 vector (x,y,z) of 2 values containing the N voxels to be used in each direction
    :return:
    '''
    # check if radius value entered is valid
    check_rad = sum([item[0]>0 for item in radius])
    if check_rad > 0:
        raise ValueError("ValueError on parameter radius: the first value of x,y,z has to be <=0")

    # Get the smaller volume around the point of interest
    x1, x2 = func_ax_indx(0, point, radius, volume)
    y1, y2 = func_ax_indx(1, point, radius, volume)
    z1, z2 = func_ax_indx(2, point, radius, volume)

    indx_x = np.linspace(x1, x2, num=x2-x1+1).astype('int')
    indx_y = np.linspace(y1, y2, num=y2-y1+1).astype('int')
    indx_z = np.linspace(z1, z2, num=z2-z1+1).astype('int')

    small_vol = volume[np.ix_(indx_x, indx_y, indx_z)]

    # Get brain label (that will serve as index)
    label_interest = int(volume[point[0], point[1], point[2]])

    return small_vol, label_interest


def get_probability(small_vol, label_interest):
    '''
    :param label_interest: label (int+, of brain region)
    :param small_vol: nxmxp overall matrix containing labels (int+)
    :return:
    '''
    n_vox_mat = small_vol.shape[0]*small_vol.shape[1]*small_vol.shape[2]
    n_vox_vect = np.bincount(small_vol.flatten().astype('int'))
    prob_region_interest = n_vox_vect[label_interest] / n_vox_mat
    return prob_region_interest, n_vox_vect, n_vox_mat


# TODO function to transform x,y,z in um to x,y,z in voxel index

# Example:
volume = np.zeros([3, 5, 5])
volume[0, :] = 23

radius = [[-2, 1], [0, 0], [-1, 1]]
point = [1, 1, 1]

small_vol, label_interest = get_small_volume(point, volume, radius)
prob_region_interest, n_vox_vect, n_vox_mat = get_probability(small_vol, label_interest)
