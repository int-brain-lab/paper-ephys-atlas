from one.webclient import http_download_file
from one.remote.aws import s3_download_file

from pathlib import Path
from one.api import ONE
import ephys_atlas.data

config = ephys_atlas.data.get_config()
GENE_EXPRESSION_PATH = Path(config['paths']['gene-expression'])
GENE_EXPRESSION_PATH.mkdir(exist_ok=True, parents=True)

files = [
    "atlas/gene-expression.pqt",
    "atlas/gene-expression.bin",
    ]


for f in files:
    s3_download_file(f, destination=GENE_EXPRESSION_PATH.joinpath(Path(f).name))


