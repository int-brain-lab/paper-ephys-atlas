"""
Module designed to work on the anatomy of the brain prior to the encoding/decoding analysis.
The EcondingAtlas is a version of the Allen Atlas relabeled to account for void labels inside of the skull compared to outside.

    ea = EncodingAtlas()
    from atlasview import atlasview
    av = atlasview.view(atlas=ea)
"""

import numpy as np
import scipy.spatial

import iblatlas.regions
from iblatlas.atlas import AllenAtlas
import iblatlas.regions

NEW_VOID = {'id': 2_000, 'name': 'void_fluid', 'acronym': 'void_fluid', 'rgb': [100, 40, 40], 'level': 0, 'parent': np.nan}


class EncodingRegions(iblatlas.regions.BrainRegions):

    def __init__(self):
        super().__init__()

    def add_new_region(self, new_region):
        """
        Adds a new region to the brain regions object
        The region will be lateralized and added at the end of the regions list
        :param new_region: dictionary with keys 'id', 'name', 'acronym', 'rgb', 'level', 'parent'
        :return: indices of the new regions
        """
        assert new_region['id'] not in self.id, 'Region ID already exists'
        order = np.max(self.order) + 1
        nr = len(self.id)
        # first update the properties of the region object, appending the new region
        self.id = np.append(self.id, [new_region['id'], -new_region['id']])
        self.name = np.append(self.name, [new_region['name'], new_region['name']])
        self.acronym = np.append(self.acronym, [new_region['acronym'], new_region['acronym']])
        self.rgb = np.append(self.rgb, np.tile(np.array(new_region['rgb'])[np.newaxis, :], [2, 1]), axis=0).astype(np.uint8)
        self.level = np.append(self.level, [new_region['level'], new_region['level']]).astype(np.uint16)
        self.parent = np.append(self.parent, [new_region['parent'], new_region['parent']])
        self.order = np.append(self.order, [order, order])
        # then need to to update the mappings and append to them as well
        for k in self.mappings:
            # if the mappign is lateralized, we need to add lateralized indices, otherwise we keep only the first
            is_lateralized = np.any(self.id[self.mappings[k]] < 0)
            inds = [nr, nr + 1] if is_lateralized else [nr, nr]
            self.mappings[k] = np.append(self.mappings[k], inds)
        return nr, nr + 1


class EncodingAtlas(AllenAtlas):
    def __init__(self):
        super().__init__()
        self.compute_surface()
        self.regions = EncodingRegions()
        self.assign_voids_inside_skull()


    def assign_voids_inside_skull(self):
        # we create a mask of the convex hull and label all of the voxels below the hull to True
        i0, i1 = np.where(~np.isnan(self.convex_top))
        i2 = self.bc.z2i(self.convex_top[i0, i1])
        mask_hull = np.zeros_like(self.label, dtype=bool)
        mask_hull[i0, i1, i2] = True
        mask_hull = np.cumsum(mask_hull, axis=2)
        # then the new voids samples are the void voxels below the convex hull
        ivoids = self.regions.add_new_region(NEW_VOID)
        # so far I am not lateralizing those voids
        self.label[np.logical_and(self.label == 0, mask_hull)] = ivoids[0]

    def compute_surface(self):
        """
        Here we compute the convex hull of the surface of the brain
        All voids below the surface are re-assigned to void_fluid new region
        :return:
        """
        super().compute_surface()
        # note that ideally we should rather take the points within the convex hull of the brain seen from the top
        iok = np.where(~np.isnan(self.top))
        yxz = np.c_[np.c_[iok], self.top[iok]]
        # computes the convex hull of the surface and interpolate over the brain
        ch = scipy.spatial.ConvexHull(yxz)
        z = scipy.interpolate.griddata(
            points=yxz[ch.vertices[1:], :2],
            values=yxz[ch.vertices[1:], 2],
            xi=yxz[:, :2], method='linear')
        # the output is the convex surface of the brain - note that we
        self.convex_top = np.zeros_like(self.top) * np.nan
        self.convex_top[iok] = z


