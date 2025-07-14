# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 09:18:43 2025

save 16-bit maps of ref channels of example session 

@author: Dinghao Luo
"""


#%% imports 
import os 
import numpy as np 
import tifffile 


#%% main
path = r'Z:\Dinghao\2p_recording\A126i\A126i-20250606\A126i-20250606-01'

binpath = os.path.join(path, 'suite2p/plane0/data.bin')
bin2path = os.path.join(path, 'suite2p/plane0/data_chan2.bin')
opspath = os.path.join(path, 'suite2p/plane0/ops.npy')

ops = np.load(opspath, allow_pickle=True).item()
tot_frames = ops['nframes']
shape = tot_frames, ops['Ly'], ops['Lx']

print('loading movies and saving references...')
mov = np.memmap(binpath, mode='r', dtype='int16', shape=shape).astype(np.float32)
mov2 = np.memmap(bin2path, mode='r', dtype='int16', shape=shape).astype(np.float32)

tot_frames = mov.shape[0]

ref1 = np.mean(mov, axis=0)
ref2 = np.mean(mov2, axis=0)

ref1_16 = ref1.astype(np.uint16)
ref2_16 = ref2.astype(np.uint16)

tifffile.imwrite(r'Z:\Dinghao\paper\figures_for_yingxue\A126i-20250606_01_ref1.tiff', ref1_16)
tifffile.imwrite(r'Z:\Dinghao\paper\figures_for_yingxue\A126i-20250606_01_ref2.tiff', ref2_16)