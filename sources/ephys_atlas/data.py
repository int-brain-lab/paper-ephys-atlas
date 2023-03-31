import logging

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
