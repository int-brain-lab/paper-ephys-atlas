# -*- coding: utf-8 -*-
"""
Created on Sun May 22 18:30:37 2022

@author: guido
"""
from os.path import join
import pandas as pd
from iblutil.numerical import ismember
from ibllib.atlas import BrainRegions
br = BrainRegions()


def load_channel_data(path):
    chan_volt = pd.read_parquet(join(path, 'channels_voltage_features.pqt'))
    chan_volt = chan_volt.drop(columns=['x', 'y', 'z'])
    mm_coord = pd.read_parquet(join(path, 'coordinates.pqt'))
    mm_coord.index.name = 'pid'
    merged_df = pd.merge(chan_volt, mm_coord, how='left', on='pid')
    merged_df = merged_df.loc[~merged_df['rms_ap'].isnull() & ~merged_df['x'].isnull()]  # remove NaNs
    
    # Remap to Beryl atlas    
    _, inds = ismember(br.acronym2id(merged_df['acronym']), br.id[br.mappings['Allen']])
    merged_df['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']
    return merged_df


def load_cluster_data(path):
    chan_volt = pd.read_parquet(join(path, 'channels_voltage_features.pqt'))
    chan_volt = chan_volt.drop(columns=['x', 'y', 'z'])
    mm_coord = pd.read_parquet(join(path, 'coordinates.pqt'))
    mm_coord.index.name = 'pid'
    merged_df = pd.merge(chan_volt, mm_coord, how='left', on='pid')
    merged_df = merged_df.loc[~merged_df['rms_ap'].isnull() & ~merged_df['x'].isnull()]  # remove NaNs
    
    # Remap to Beryl atlas    
    _, inds = ismember(br.acronym2id(merged_df['acronym']), br.id[br.mappings['Allen']])
    merged_df['beryl_acronyms'] = br.get(br.id[br.mappings['Beryl'][inds]])['acronym']
    return merged_df