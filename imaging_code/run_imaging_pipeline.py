# -*- coding: utf-8 -*-
"""
Created on Mon 13 May 17:13:42 2024
Modified on Tue 25 June 15:09:40 2024

This code combines grid_extract.py (Dinghao) and after_suite2p.py (Jingyu)

@authors: Dinghao Luo, Jingyu Cao
@modifiers: Dinghao Luo, Jingyu Cao
"""


#%% imports 
import sys
import pandas as pd 
from time import time
from datetime import timedelta

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf
import imaging_pipeline_main_functions as ipmf


#%% recording lists 
sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE

sys.path.append("Z:\Jingyu\Code\Python")
# import anm_list_running
# path_dLight = anm_list_running.grid_tmp


#%% suite2p ROIs or grid ROIs
roi_switch = input('process with...\n1: suite2p ROIs\n2: grid ROIs\n')
if roi_switch not in ['1', '2']:
    print('\nWARNING:\ninvalid input; halting processing')


#%% customise parameters
# plot reference or not 
plot_ref = True

# align to...
align_run = 1
align_rew = 1
align_cue = 0

# smoothing
smooth = 1

# how many seconds before and after each align_to landmark 
bef = 1; aft = 4  # in seconds 

if roi_switch=='1':
    # do we want to calculate dFF
    dFF = 1
    plot_heatmap = 1 
    plot_trace = 1 
    
if roi_switch=='2':
    # how many pixels x/y for each grid
    stride = int(496/2/2/2/2)
    # stride = 496
    border = 8  # ignore how many pixels at the border (1 side)
    # border = 6
    fit = ipf.check_stride_border(stride, border)
    
    dFF = 1
    
    # do we want to save the grid_traces if extracted 
    save_grids = 1
    
    print('\n')


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


#%% print out 
if roi_switch=='1': 
    roi_string='suite2p'
    printout = """
    processing {} ROIs...
        align_run = {}
        align_rew = {}
        align_cue = {}
        bef = {}
        aft = {}
        smooth = {}
        """.format(roi_string, align_run, align_rew, align_cue, bef, aft, smooth)
if roi_switch=='2':
    roi_string='grid'
    printout = """
    processing {} ROIs...
        align_run = {}
        align_rew = {}
        align_cue = {}
        bef = {}
        aft = {}
        stride = {}
        border = {}
        smooth = {}
        dFF = {}
        """.format(roi_string, align_run, align_rew, align_cue, bef, aft, stride, border, smooth, dFF)
print(printout)


#%% run
print('loading behaviour dataframe...')
try:
    df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\behaviour\all_GRABNE_sessions.pkl')
except FileNotFoundError:
    print('loading failed: no behavioural dataframe found\n')

for rec_path in pathGRABNE[102:]:
    if 'Dinghao' in rec_path:
        reg_path = rec_path+r'\processed\suite2p\plane0'
        recname = rec_path[-17:]
        txt_path = r'Z:\Dinghao\MiceExp\ANMD{}\{}{}T.txt'.format(recname[1:4], recname[:4], recname[5:])
    if 'Jingyu' in rec_path:
        reg_path = rec_path+r'\RegOnly\suite2p\plane0'
        recname = rec_path[-17:-3]+'-'+rec_path[-2:]
        txt_path = r'Z:\Jingyu\mice-expdata\{}\A{}T.txt'.format(rec_path[-23:-18], recname[2:])
    
    print('\nprocessing {}'.format(recname))
    t0 = time()
    
    try:
        print('reading session behavioural data...')
        beh = df.loc[recname]
    except NameError:
        beh = []
    
    # main calls
    if roi_switch=='1':
        ipmf.run_suite2p_pipeline(rec_path, recname, reg_path, txt_path,
                                  plot_ref, plot_heatmap, plot_trace,
                                  smooth, dFF,
                                  bef, aft,
                                  align_run, align_rew, align_cue)
    if roi_switch=='2':
        ipmf.run_grid_pipeline(rec_path, recname, reg_path, txt_path, beh,
                               stride, border, 
                               plot_ref, 
                               smooth, dFF, save_grids, 
                               bef, aft,
                               align_run, align_rew, align_cue,
                               GPU_AVAILABLE)
        
    print('{} done ({})'.format(recname, str(timedelta(seconds=int(time()-t0)))))