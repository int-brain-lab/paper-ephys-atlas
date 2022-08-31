# paper-ephys-atlas
Code related to the electrophysiology atlas paper

## Getting started

To get the features dataframe needed to train models, see [the loading example](sources/examples/00_load_clusters_tables.py)

To get a naive example of decoding model out of ephys features, see [a raw ephys decoder](sources/decoding/raw_ephys_decodes_regions.py)


## Install instructions
Clone the repository
```bash
git clone https://github.com/int-brain-lab/paper-ephys-atlas.git
```

Activate your environment of choice (usually `iblenv` as described here: https://github.com/int-brain-lab/iblenv).
```bash
conda activate iblenv
```

Cd into the repository and install in-place
```bash
cd paper-ephys-atlas
pip install -e .
```


