# -*- coding: utf-8 -*-
"""
Created on 4 June 16:30:51 2024

**NOTE: THIS SCRIPT RUNS THE CUSTOMISED SUITE2P-WANG-LAB INSTEAD OF SUITE2P**
**SUITE2P-WANG-LAB CAN BE ACCESSED HERE: 
    https://github.com/the-wang-lab/suite2p-wang-lab**

@author: Dinghao Luo
    - merged registration into script as a function
    - completely revamped main block to read in rec_list
        - if needs be, can import a separate list of parameters fit for each 
          recording session
    - changed the main run function 

"""


#%% imports 
import sys
import suite2p
import os
import numpy as np
from contextlib import redirect_stdout
import shutil

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathGRABNE = rec_list.pathHPCGRABNE


#%% functions
def register(path):
    
    # timer 
    from time import time
    from datetime import timedelta
    
    # load parameter files
    ops = np.load(r'Z:/Dinghao/2p_recording/registration_parameters.npy', allow_pickle=True).item()
    
    # customise parameters
    ops['input_format'] = 'tif'
    ops['sparse_mode'] = True
    ops['roidetect'] = False
    ops['reg_tif'] = True
    ops['reg_tif_chan2'] = True
    # ops['anatomical_only'] = False
    # ops['spatial_scale'] = True
    # ops['save_roi_iterations'] = True
    # ops['do_extraction'] = False
    
    outdir = path + r'\registered'
    os.makedirs(outdir, exist_ok=True)
    ops['save_path0'] = outdir
    
    print(r'registration starts (check \registered for progress)')
    t0 = time()

    # save text output
    pathlog = outdir+r'/run_suite2p-wang-lab.log'

    # run suite2p
    db = {
        'data_path': [str(path)],
        'save_path0': outdir
        }
        
    with open(pathlog, 'w') as f:
        with redirect_stdout(f):
            print('running suite2p v{} from Spyder'.format(suite2p.version))
            suite2p.run_s2p(ops=ops, db=db)
            
    print('registration complete ({})\n'.format(str(timedelta(seconds=time()-t0))))
            

def copy_suite2p_files(s, data, mode, ori_mode = 'RegOnly'):
    p_ori = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:], ori_mode, 'suite2p', 'plane0',data)
    p_des = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:], mode, 'suite2p', 'plane0')
    p_new = os.path.join(p_des,data)
    os.makedirs(p_des, exist_ok=True)
    shutil.copy(p_ori, p_new)


def run(path, mode):
    p_data = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:])
    p_out = os.path.join(p_data, mode)

    ops = np.load(os.path.join(p_data, mode, 'suite2p', 'plane0','ops.npy'), allow_pickle=True).item()
    ops['do_registration']=1
    # ops["input_format"]=="binary"
    ops['sparse_mode'] = True
    ops['anatomical_only'] = 0
    ops['roidetect'] = 1
    ops['spatial_scale']=2
    ops['denoise']=1
    
    
    #-----------for neuropil extraction--------------
    ops['circular_neuropil'] = True
    ops['inner_neuropil_radius']=5
    #------------------------------------------------
    

    ops['max_iterations'] = 1 #max_iterations=250 * ops["max_iterations"]
    ops['high_pass']=200
    
    ops['wang:bin_size'] = 1
    ops['wang:high_pass_overlapping']=True
    ops['wang:rolling_width']=30
    ops["wang:rolling_bin"] = 'max' #*******************************
    ops['wang:use_alt_norm']=True
    ops['wang:downsample_scale'] = 1
    ops['wang:thresh_act_pix']=0.04#0.74 #lam.max*'thresh_act_pix'] 
    # (thresh_lam = max(0, lam.max()) * thresh_active...
    # (pix_act = lam > thresh_lam
    ops['wang:thresh_peak_default']= 0.03 #0.085 #default:.08
    # ops['spatial_scale'] = 1

    ops['wang:save_roi_iterations']=True
    ops['save_path_new'] = p_out #****************************


    # if os.path.exists(p_out) == False:
    os.makedirs(p_out, exist_ok=True)
    # ops['path_roi_iterations'] = p_outops['save_path0']
    ops['wang:save_path_sparsedetect'] = p_out
    ops['wang:neuropil_lam']= True#****************************
    ops['wang:movie_chunk'] = 10000 #whether to chunck movie for ROI detection, frames used
    ops['wang:norm_method'] = 'max'


    print(f'INFO: Running suite2p-wang-lab for {p_data}')

    # save output to 'run.log' file
    os.makedirs(p_out, exist_ok = True)
    p_log = p_out+r'/run_suite2p-wang-lab.log'    
    print(f'INFO: Saving text output to {p_log}')

    # run suite2p
    db = {
        'data_path': [ str(p_data) ],
        'save_path0': p_out,
        # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
    }

    with open(p_log, 'w') as f:
        with redirect_stdout(f):
            print(f'Running suite2p v{suite2p.version} from Spyder')
            suite2p.run_s2p(ops=ops, db=db)

    print(f'{s}------Finished------')


#%% run all sessions
for path in pathGRABNE:
    sessname = path[-17:]
    print(sessname)
    
    reg_path = path+r'\registered'
    if not os.path.exists(reg_path):  # if registration has not been performed
        register(path)
    # if 'no tiffs' is raised, most likely it is due to typos in pathnames
    
    # run(path, mode)
    
    # print(f"----copying-{s}----")
    # ops = np.load(os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:], mode, 'suite2p', 'plane0','ops.npy'), allow_pickle=True).item()
    # new_mode = 'denoise={}_rolling={}_pix={}_peak={}_iterations={}_norm={}_neuropillam={}'.format(ops['denoise'], 
    #                                                                                                   ops["wang:rolling_bin"],
    #                                                                                                   ops['wang:thresh_act_pix'],
    #                                                                                                   ops['wang:thresh_peak_default'],
    #                                                                                                   ops['max_iterations'],
    #                                                                                                   ops['wang:norm_method'],
    #                                                                                                   ops['wang:neuropil_lam'])   
    # for i in ('ops.npy', 'F.npy', 'F_chan2.npy', 'Fneu.npy', 'Fneu_chan2.npy', 'iscell.npy', 'redcell.npy', 'spks.npy', 'stat.npy' ):
    #     copy_suite2p_files(s, i, new_mode, ori_mode = 'RegOnly')
    # #copy run-log file
    # p_ori = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:], mode, 'run_suite2p-wang-lab.log')
    # p_new = os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:], new_mode, 'run_suite2p-wang-lab.log')
    # shutil.copy(p_ori, p_new)
    # print("----finished----")
    
    # # check whether ROI detection has been finished
    # check_mode = 'denoise=1_rolling=max_pix=0.04_peak=0.03_iterations=1_norm=max_neuropillam=True'
    # if os.path.exists(os.path.join(r'Z:\Jingyu\2P_Recording',s[0:-12], s[:-3], s[-2:], check_mode, 'suite2p', 'plane0', 'stat.npy')):
    #     print(f'***{s}_ROI detection finished already***')