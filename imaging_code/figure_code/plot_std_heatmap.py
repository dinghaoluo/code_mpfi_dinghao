# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:14:17 2024

standard deviation map of session to compare local/global signal fluctuations

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('cupy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print('An error occurred: {}'.format(e))
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print('GPU-acceleration with {} and cupy'.format(str(name)))
else:
    print('GPU-acceleartion unavailable')


#%% process
# reg_path = r'Z:\Jingyu\2P_Recording\AC950\AC950-20240820\02\RegOnly\suite2p\plane0'
# reg_path = r'Z:\Dinghao\2p_recording\A094i\A094i-20240716\A094i-20240716-01\processed\suite2p\plane0'
reg_path = r'Z:\Jingyu\2P_Recording\AC955\AC955-20240911\02\RegOnly\suite2p\plane0'
recname = r'AC955-20240911'

opsfile = reg_path+r'\ops.npy'
ops = np.load(opsfile, allow_pickle=True).item()
tot_frames = ops['nframes']
shape = tot_frames, ops['Ly'], ops['Lx']

binfile = reg_path+r'\data.bin'
data = np.memmap(binfile, dtype='int16', mode='r', shape=shape)[:2000]

# calculate dFF
from time import time 
from datetime import timedelta

t0 = time()
dFF = ipf.calculate_dFF(data, t_axis=0, GPU_AVAILABLE=GPU_AVAILABLE)

print(f'dFF calculation finished... ({timedelta(seconds=int(time()-t0))})')

t0 = time()
pixel_std = np.std(dFF, axis=0)

print(f'std calculation finished... ({timedelta(seconds=int(time()-t0))})')


#%% plotting
# plot the standard deviation map
if 'Jingyu' in reg_path: sensor = 'dLight'
if '955' in reg_path or '092' in reg_path: sensor = 'GRABDA'
if 'Dinghao' in reg_path: sensor = 'GRABNE'

fig, ax = plt.subplots(figsize=(3,3))
im_dlight = ax.imshow(pixel_std, cmap='magma', interpolation=None, vmin=0, vmax=7)
plt.colorbar(im_dlight, label='std. of dF/F', shrink=.5)
ax.set(xticks=[], yticks=[],
       title=f'{sensor} std. map (dF/F)')

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\{}\{}_std_map_{}{}'.format(sensor, sensor, recname, ext),
                dpi=300)