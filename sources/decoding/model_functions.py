# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:30:37 2022

@author: guido
"""
from os.path import join, dirname, realpath, split
import pandas as pd
from joblib import load
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
br = BrainRegions()


def load_channel_data():
    # Load in data
    path = join(split(dirname(realpath(__file__)))[0], 'training_data')
    chan_volt = pd.read_parquet(join(path, 'channels_voltage_features.pqt'))
    chan_volt = chan_volt.drop(columns=['x', 'y', 'z'])
    
    # Add micromanipulator coordinates
    mm_coord = pd.read_parquet(join(path, 'coordinates.pqt'))
    mm_coord.index.name = 'pid'
    merged_df = pd.merge(chan_volt, mm_coord, how='left', on='pid')
    
    # remove NaNs
    merged_df = merged_df.loc[~merged_df['rms_ap'].isnull() & ~merged_df['x'].isnull()]  
    
    # Remap to Beryl atlas    
    _, inds = ismember(br.acronym2id(merged_df['acronym']), br.id[br.mappings['Allen']])
    merged_df['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']
    
    # Remap to Cosmos atlas    
    _, inds = ismember(br.acronym2id(merged_df['acronym']), br.id[br.mappings['Allen']])
    merged_df['cosmos_acronyms'] = br.get(br.id[br.mappings['Cosmos'][inds]])['acronym']
    merged_df = merged_df.rename({'acronym': 'allen_acronyms'}, axis=1)
    
    return merged_df


def load_trained_model(model='channels', atlas='beryl'):
    """
    Parameters
    ----------
    model : str
        'channels' or 'clusters'
    """
    path = join(split(dirname(realpath(__file__)))[0], 'trained_models')
    clf = load(join(path, f'{model}_model_{atlas}.pkl'))
    return clf


def load_cluster_data():
    # Load in data
    path = join(split(dirname(realpath(__file__)))[0], 'training_data')
    df_clusters = pd.read_parquet(join(path, 'clusters.pqt'))
    df_channels = pd.read_parquet(join(path, 'channels.pqt'))
        
    # Add acronyms
    if 'atlas_id' not in df_clusters.keys():
        df_clusters = df_clusters.merge(
            df_channels[['atlas_id', 'acronym']], right_on=['pid', 'raw_ind'], left_on=['pid', 'channels'])
       
    # Remap to Beryl atlas    
    _, inds = ismember(br.acronym2id(df_clusters['acronym']), br.id[br.mappings['Allen']])
    df_clusters['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']
        
    # remove NaNs
    df_clusters = df_clusters.loc[~df_clusters['amp_max'].isnull()]  
    
    return df_clusters