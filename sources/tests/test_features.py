import numpy as np
import ephys_atlas.features

file_numpy = "/Users/olivier/Documents/datadisk/Data/paper-ephys-atlas/ephys-atlas-sample/749cb2b7-e57e-4453-a794-f6230e4d0226/T02500/lf.npy"
data_lf = np.load(file_numpy).astype(np.float32)

fs = 250.00203042940993

df_chunk = ephys_atlas.features.lf(data_lf, fs=fs)
