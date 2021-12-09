def bwm_pids(one):
    django_strg = ['session__project__name__icontains,ibl_neuropixel_brainwide_01',
                   'session__qc__lt,50',
                   '~json__qc,CRITICAL',
                   #'json__extended_qc__tracing_exists,True',  # TODO remove ?
                   'session__extended_qc__behavior,1,'
                   'session__json__IS_MOCK,False']

    insertions = one.alyx.rest('insertions', 'list', django=django_strg)
    return [item['id'] for item in insertions], insertions

