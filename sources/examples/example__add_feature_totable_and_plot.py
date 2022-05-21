'''
0. Look at the features already existing https://drive.google.com/drive/u/1/folders/1y2uaiyYnWqVJqtqMv4FxxFQ_8LT2W6xs
1. Define compute function
2. Download needed data (run example on 1 insertion)
3. Launch computation in loop
4. Append to dataframe ; save to dataframe
'''
from one.api import ONE
from ephys_atlas.data import bwm_pids

# ==== INIT

one = ONE()

excludes = [
    # 'd8c7d3f2-f8e7-451d-bca1-7800f4ef52ed',  # key error in loading histology from json
    # 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',  # same same
    # 'c2184312-2421-492b-bbee-e8c8e982e49e',  # same same
    # '58b271d5-f728-4de8-b2ae-51908931247c',  # same same
    'f86e9571-63ff-4116-9c40-aa44d57d2da9',  # 404 not found
    '16ad5eef-3fa6-4c75-9296-29bf40c5cfaa',  # 404 not found
    '511afaa5-fdc4-4166-b4c0-4629ec5e652e',  # 404 not found
    'f88d4dd4-ccd7-400e-9035-fa00be3bcfa8',  # 404 not found
]
pids, _ = bwm_pids(one, tracing=True)

# ==== Step 1

def fanofactor():
    # bla

# ==== Step 2

