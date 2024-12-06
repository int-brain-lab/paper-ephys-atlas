'''
Script to examplify how to upload
- vlues per brain region (e.g. mean nof feature)
- values on dots (e.g. channels, neurons) displayed volumetrically
- volume of values

'''
##
# Init bucket
import numpy as np
from iblbrainviewer.api import FeatureUploader

bucket = 'gc_volume_bucket'
bucket = 'ow_bucket'

# Create or load the bucket.
up = FeatureUploader(bucket)
##
# Upload features
# cf documentation:
# https://github.com/int-brain-lab/iblbrainviewer/blob/9be77aac83cfe42af32131d5dbf30493b3c9a774/docs/atlas_website_getting_started.ipynb

##
# Upload dots

fname = 'mydots'
tree = {'my test dots': fname}

# Create the features.
short_desc = "these are my dots"
n = 10000
xyz = np.random.normal(scale=1e-3, size=(n, 3)).astype(np.float32)
values = np.random.uniform(low=0, high=1, size=(n,)).astype(np.float32)
up.upload_dots(fname, xyz, values, short_desc=short_desc, patch=up.features_exist(fname))

##
# Upload volume
# The shape must be exactly:  (528, 320, 456)
fname = 'my_volume'
volume = np.random.uniform(low=0, high=1, size=(528, 320, 456)).astype(np.float32)
up.upload_volume(fname, volume, min_max=None, short_desc=None, patch=False)
