"""
Find how many sessions / insertions have associated PASSIVE dataset
"""

import numpy as np
from one.api import ONE

one = ONE()

django_strg = [
    "session__projects__name__icontains,ibl_neuropixel_brainwide_01",
    "session__qc__lt,50",
    "~json__qc,CRITICAL",
    "session__extended_qc__behavior,1," "session__json__IS_MOCK,False",
]

insertions = one.alyx.rest(
    "insertions", "list", django=django_strg
)  # 603 on 21-06-2022

# get session eid as dataset associated to this
sess_overall_good = np.unique([item["session"] for item in insertions])


# Check for specific datasets
def get_sess_missingds(ds_spec, sess_overall_good):
    sess_ds = one.alyx.rest(
        "sessions",
        "list",
        task_protocol="ephys",
        projects="ibl_neuropixel_brainwide_01",
        dataset_types=ds_spec,
    )
    sess_ds_id = [item["url"][-36:] for item in sess_ds]

    a = set(sess_overall_good)
    b = set(sess_ds_id)
    c = a.difference(b)
    return c, b


ds_raw = ["_iblrig_RFMapStim.raw"]

ds_ext = [
    "_ibl_passivePeriods.intervalsTable",
    "_ibl_passiveRFM.times",
    "_ibl_passiveGabor.table",
    "_ibl_passiveStims.table",
]

# make set
sess_raw_ds_miss, sess_raw_have = get_sess_missingds(ds_raw, sess_overall_good)
sess_ext_ds_miss, sess_ext_have = get_sess_missingds(ds_ext, sess_overall_good)

# Can be extracted
sess_to_extract = sess_ext_ds_miss.difference(sess_raw_have)  # 60 sess on 21-06-2022

# How many insertions does this relates to
pid_ext_have = [
    item["id"] for item in insertions if item["session"] in sess_ext_have
]  # 475 pid on 21-06-2022
pid_to_extract = [
    item["id"] for item in insertions if item["session"] in sess_to_extract
]  # 92 pid on 21-06-2022
# 603 - 567 = 36 insertions will never have passive CW on 21-06-2022
