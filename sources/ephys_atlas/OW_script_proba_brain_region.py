'''
Written by Olivier
'''

import numpy as np
from neurodsp.utils import fcn_cosine
from ibllib.atlas import AllenAtlas

ba = AllenAtlas()
volume = ba.label

fcc = fcn_cosine([50, 150])

nx, ny, nz = (9, 11, 7)

point_um = [0, 0, - 800 * 1e-6]
pt = ba.bc.xyz2i(point_um)

xp = (np.arange(nx) - np.mean(np.arange(nx))) + pt[0]
yp = (np.arange(ny) - np.mean(np.arange(ny))) + pt[1]
zp = (np.arange(nz) - np.mean(np.arange(nz))) + pt[2]

xp = xp[np.where((xp>=0) & (xp<volume.shape[0]))].astype('int')
yp = yp[np.where((yp>=0) & (yp<volume.shape[0]))].astype('int')
zp = zp[np.where((zp>=0) & (zp<volume.shape[0]))].astype('int')


small_vol = volume[np.ix_(*[(xp, yp, zp)[i] for i in ba.xyz2dims])]




x = (np.arange(nx) - np.mean(np.arange(nx))) * ba.bc.dx * 1e6
y = (np.arange(ny) - np.mean(np.arange(ny))) * ba.bc.dy * 1e6
z = (np.arange(nz) - np.mean(np.arange(nz))) * ba.bc.dz * 1e6

X, Y, Z = np.meshgrid(x, y , z)
radius = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

wei = 1 - fcc(radius)


n_vox_vect = np.bincount(small_vol.flatten().astype('int'), weights=wei.flatten())

n_vox_vect /= np.sum(n_vox_vect)

mapping = 'Beryl'
iregions = np.where(n_vox_vect)[0]
proba_regions = n_vox_vect[iregions]
reg = ba.regions.mappings[mapping][iregions]

unik_reg = np.unique(reg)
prob = np.zeros([unik_reg.size])



for n, i in enumerate(unik_reg):
    prob[n] = np.sum(proba_regions[np.where(reg == i)[0]])
