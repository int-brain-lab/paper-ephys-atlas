from pathlib import Path
from one.remote import aws
from one.api import ONE

LOCAL_DATA_PATH = Path("/Users/olivier/Documents/datadisk/atlas")

# The AWS private credentials are stored in Alyx, so that only one authentication is required
one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='online')
s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
aws.s3_download_folder("data/tables/atlas",
                       LOCAL_DATA_PATH,
                       s3=s3, bucket_name=bucket_name)