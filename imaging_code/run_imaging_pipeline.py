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
from time import time
from datetime import timedelta

if r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils' not in sys.path:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import imaging_pipeline_functions as ipf
import imaging_pipeline_main_functions as ipmf


#%% recording lists 
if r'Z:\Dinghao\code_dinghao' not in sys.path:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE

if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
import anm_list_running
path_dLight = anm_list_running.grid_tmp


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
    # stride = int(496/2/2/2/2)
    stride = 496
    border = 8  # ignore how many pixels at the border (1 side)
    # border = 6
    fit = ipf.check_stride_border(stride, border)
    
    dFF = 1
    
    # do we want to save the grid_traces if extracted 
    save_grids = 1


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
for rec_path in pathGRABNE:
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
    
    # main calls
    if roi_switch=='1':
        ipmf.run_suite2p_pipeline(rec_path, recname, reg_path, txt_path,
                                  plot_ref, plot_heatmap, plot_trace,
                                  smooth, dFF,
                                  bef, aft,
                                  align_run, align_rew, align_cue)
    if roi_switch=='2':
        ipmf.run_grid_pipeline(rec_path, recname, reg_path, txt_path, 
                               stride, border, 
                               plot_ref, 
                               smooth, dFF, save_grids, 
                               bef, aft,
                               align_run, align_rew, align_cue)
        
    print('{} done ({})'.format(recname, str(timedelta(seconds=int(time()-t0)))))