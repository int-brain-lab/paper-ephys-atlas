# paper-ephys-atlas
Code related to the electrophysiology atlas paper

## Getting started

To get the features dataframe needed to train models, see [the loading example](sources/examples/00_load_clusters_tables.py)

To get a naive example of decoding model out of ephys features, see [a raw ephys decoder](sources/decoding/raw_ephys_decodes_regions.py)


## Configuration instructions
The configuration file is located in `config-ephys-atlas.yaml`.
It contains local data storage paths and some parameters for the analysis. To avoid committing changes to this file, it is ignored by git. To make changes to the configuration, copy the template file and edit the copy:

```bash
cp config-ephys-atlas.template.yaml config-ephys-atlas.yaml
```


## Install instructions
Clone the repository
```bash
git clone https://github.com/int-brain-lab/paper-ephys-atlas.git
```

Activate your environment of choice (usually a conda `iblenv` as described here: https://github.com/int-brain-lab/iblenv).
```bash
conda activate iblenv
```

Cd into the repository and install in-place
```bash
cd paper-ephys-atlas
pip install -e .
```


## Download the features table
The features table is stored with our aggregates dataset as part of IBL data. To download it, run the following commands in Python:
```python
from pathlib import Path
from one.api import ONE
import ephys_atlas.data

LOCAL_DATA_PATH = Path.home().joinpath("Downloads")
LABEL = "2024_W50"  # or put "latest"
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='local')
df_raw_features, df_clusters, df_channels, df_probes = ephys_atlas.data.download_tables(label=LABEL, local_path=LOCAL_DATA_PATH, one=one)
```
