'''
Get autism data from JP
fmr1 mouse line
'''
from one.api import ONE

one = ONE(mode='remote')
project = 'angelaki_mouseASD'
mouse_line = 'B6.129P2-Fmr1<tm1Cgr>/J'
# Note: not all FMR mice are labelled with line on database ==> use name instead
# https://alyx.internationalbrainlab.org/admin/subjects/subject/?alive=n&q=FMR&responsible_user=all

# Get all insertions for this project
str_query = f'session__project__name__icontains,{project},' \
             'session__qc__lt,50,' \
             '~json__qc,CRITICAL'
insertions = one.alyx.rest('insertions', 'list', django=str_query)
# Restrict to only those with subject starting with FMR
ins_keep = [item for item in insertions if item['session_info']['subject'][0:3] == 'FMR']

# Print
for ins in ins_keep:
    print(f"PID: {ins['id']} - "
          f"{ins['session_info']['subject']}/{ins['session_info']['start_time'][0:10]}/{ins['session_info']['number']}")
