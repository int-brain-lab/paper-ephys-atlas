import pandas as pd
import numpy as np

from pathlib import Path
from one.remote import aws
from one.api import ONE

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from random import shuffle

def get_data():
    LOCAL_DATA_PATH = Path(
        "/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/atlas_data")

    one = ONE(base_url="https://alyx.internationalbrainlab.org", mode='online')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_folder("aggregates/bwm",
                           LOCAL_DATA_PATH,
                           s3=s3, bucket_name=bucket_name)

    df_clusters = pd.read_parquet(LOCAL_DATA_PATH.joinpath('clusters.pqt'))
    df_probes = pd.read_parquet(LOCAL_DATA_PATH.joinpath('probes.pqt'))
    df_channels = pd.read_parquet(LOCAL_DATA_PATH.joinpath('channels.pqt'))
    df_depths = pd.read_parquet(LOCAL_DATA_PATH.joinpath('depths.pqt'))
    df_voltage = pd.read_parquet(LOCAL_DATA_PATH.joinpath('raw_ephys_features.pqt'))

    df_voltage = pd.merge(df_voltage, df_channels, 
                 left_index=True, right_index=True).dropna()

    return df_voltage


def regress(scaling = False, shuf = False):

    df_voltage = get_data()
    
    x_list = ['rms_ap', 'alpha_mean', 'alpha_std', 'spike_rate', 
              'cloud_x_std', 'cloud_y_std', 'cloud_z_std', 'rms_lf', 
              'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta', 'psd_gamma']

    X = df_voltage.loc[:, x_list].values
    y = df_voltage.loc[:, ['x','y','z']].values

    print(X.shape, 'samples x input_features')
    print(y.shape, 'samples x target_features')
    

    if scaling:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if shuf:
        shuffle(y)

    # cross validation    
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True)
    fold = 0
    
    for tra, tes in kf.split(X):

        X_tra = X[tra]
        X_tes = X[tes]
        y_tra = y[tra] 
        y_tes = y[tes]         
        
        reg = LinearRegression().fit(X_tra, y_tra)
        print('fold: ', fold, 'score: ', reg.score(X_tes, y_tes))
        fold += 1 


