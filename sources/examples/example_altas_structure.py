from ibllib.atlas.regions import BrainRegions
from ibllib.atlas import AllenAtlas
import numpy as np
"""
The Allen atlas organisation has 3 main components, see the
atlas structure table: https://github.com/int-brain-lab/ibllib/blob/master/ibllib/atlas/allen_structure_tree.csv

1.  The row index 0-1-2-3...-1327 of the table (on github, it start with 'void' on i_row = 2 ;
    but you should think it starts at 0). This is called "label" in the IBL atlas code below.
2.  The Allen ID, e.g. 0-997-8 ... (see column 'id' in the table) ; this is a unique identifier number per brain region
    entry in the Allen atlas
3.  The acronym of the brain region, e.g. void-root-... (see column 'name' in the table)
"""


## Instantiate the atlas volume (3D matrix), resolution of 25um
ba = AllenAtlas(25)

# The greyscale value for each voxel (used to form the anatomical image) is stored into the `image` attribute.
grey_scale_img = ba.image
grey_scale_img.shape  # The shape of it is (528, 456, 320)

# The *row indices* of the brain regions for each voxel are stored into the `label` attribute.
# Note that these are *NOT* the same as the ID values of each Allen brain region
rowindx = ba.label
rowindx.shape  # The shape of it is also (528, 456, 320)

# Note that values are always stored in a rectangular (528, 456, 320) matrix, but the brain is a ball in the middle.
# To find which voxels in the (528, 456, 320) matrix are actual brain, filter for the acronym "void"
# For this, you need to instantiate the BrainRegions to load in the acronyms
br = BrainRegions()

# The Allen ID and acronyms are stored in `id` and `acronym`:
# Note : the [0], [1] here corresponds to the label , i.e. "i_row" in the atlas structure
br.id[0]  # = 0, corresponding to 'void'
br.id[1]  # = 997 , corresponding to 'root'
br.acronym[0]  # = 'void'
br.acronym[1]  # = 'root'

rindx_void = np.where(br.acronym == 'void')
np.where(rowindx != rindx_void)
"""
This give out an array:
(array([  1,   1,   1, ..., 526, 526, 526]),
 array([175, 175, 175, ..., 318, 318, 318]),
 array([187, 188, 189, ..., 153, 154, 155]))
"""

# To find what is the brain region acronym associated to a given voxel (e.g. coordinates index [1,175,187] ):
rowindx[1,175,187]  # Get the corresponding label, or "i_row", value ; = 2434
br.acronym[2434]  # Get the corresponding acronym from that label using the BrainRegions variable ; = 'onl'
"""
Note on the value 2434:
The acronym table index: https://github.com/int-brain-lab/ibllib/blob/master/ibllib/atlas/allen_structure_tree.csv
ranges from 0-1327 (see N rows)
The 'onl' acronym is on row index 1107
To differentiate right and left hemisphere, these rows are actually doubled in the code (-> size 2655)
(and assigned a ± sign depending on the hemisphere)

So index  1107  , and   1327 + 1107 = 2434   will both refer to 'onl' but each in left/right hemisphere.
"""

# To get the ID/acronym of the brain region as per the Allen atlas (0 for void, 997 for root etc) into the 3D volume:
mapped_atlas_id = br.id[ba.label]
mapped_atlas_ac = br.acronym[ba.label]
# This can ease computation when looking for a given brain region, e.g. for root:
np.where(mapped_atlas_id == 997)

# To reorganise the mapping, use the br.mappings function:
cosmos_label = br.mappings['Cosmos']
# This is a 2655 vector containing the unique values:
# [   0,    1,    6,  380,  455,  556,  571,  642,  716,  807,  883,  1015]
# which correspond to :
br.acronym[np.unique(br.mappings['Cosmos'])]
# ['void', 'root', 'Isocortex', 'OLF', 'HPF', 'CTXsp', 'CNU', 'TH', 'HY', 'MB', 'HB', 'CB']
