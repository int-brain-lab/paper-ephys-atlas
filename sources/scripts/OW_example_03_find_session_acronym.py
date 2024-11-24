"""
Search for data (session/insertion) given a specific brain region
"""
# Note: there could be edge cases where having an insertion with channels inside a regions may not translete
# in clusters in the region: if the probe has only a few channels in the said region, and the channels don't
# have spikes that may be the case

from one.api import ONE

one = ONE()

acronym = "CA1"  # The acronym does *not* need to be a leaf node

sessions = one.alyx.rest(
    "sessions",
    "list",
    task_protocol="ephys",
    atlas_acronym=acronym,
    project="ibl_neuropixel_brainwide",
    no_cache=True,
)


insertions = one.alyx.rest(
    "insertions",
    "list",
    task_protocol="ephys",
    atlas_acronym=acronym,
    project="ibl_neuropixel_brainwide",
    no_cache=True,
)
