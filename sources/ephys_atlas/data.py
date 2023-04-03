import logging
from pathlib import Path

import pandas as pd

from one.remote import aws

_logger = logging.getLogger('ibllib')

SPIKES_ATTRIBUTES = ['clusters', 'times', 'depths', 'amps']
CLUSTERS_ATTRIBUTES = ['channels', 'depths', 'metrics']


def atlas_pids(one, tracing=True):
    django_strg = [
        'session__project__name__icontains,ibl_neuropixel_brainwide_01',
        'session__qc__lt,50',
        '~json__qc,CRITICAL',
        # 'session__extended_qc__behavior,1',
        'session__json__IS_MOCK,False',
    ]
    if tracing:
        django_strg.append('json__extended_qc__tracing_exists,True')

    insertions = one.alyx.rest('insertions', 'list', django=django_strg)
    return [item['id'] for item in insertions], insertions


def download_tables(local_path, label='2022_W34', one=None, verify=True):
    # The AWS private credentials are stored in Alyx, so that only one authentication is required
    local_path = Path(local_path)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_folder(f"aggregates/atlas/{label}",
                           local_path,
                           s3=s3, bucket_name=bucket_name)

    df_clusters = pd.read_parquet(local_path.joinpath('clusters.pqt'))
    df_channels = pd.read_parquet(local_path.joinpath('channels.pqt'))
    df_voltage = pd.read_parquet(local_path.joinpath('raw_ephys_features.pqt'))
    if verify:
        verify_tables(df_voltage, df_clusters, df_channels)
    return df_voltage, df_clusters, df_channels


def verify_tables(df_voltage, df_clusters, df_channels):
    """
    Verify that the tables have the correct format and indices
    :param df_clusters:
    :param df_channels:
    :param df_voltage:
    :return:
    """
    assert df_clusters.index.names == ['pid', 'cluster_id']
    assert df_channels.index.names == ['pid', 'channel']
    assert df_voltage.index.names == ['pid', 'channel']
