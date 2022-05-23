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
    return merged_df


def load_trained_model(model):
    """
    Parameters
    ----------
    model : str
        'channels' or 'clusters'
    """
    path = join(split(dirname(realpath(__file__)))[0], 'trained_models')
    clf = load(join(path, f'{model}_model.pkl'))
    return clf


def load_cluster_data():
    # Load in data
    path = join(split(dirname(realpath(__file__)))[0], 'training_data')
    clusters = pd.read_parquet(join(path, 'clusters.pqt'))
   
    # Remap to Beryl atlas    
    _, inds = ismember(br.acronym2id(clusters['acronym']), br.id[br.mappings['Allen']])
    clusters['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']
    return clusters