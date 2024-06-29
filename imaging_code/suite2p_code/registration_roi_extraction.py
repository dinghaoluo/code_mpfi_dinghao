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
    
    outdir = path + r'\processed'
    os.makedirs(outdir, exist_ok=True)
    ops['save_path0'] = outdir
    
    print(r'registration starts (check \processed for progress)')
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
            
    print('registration complete ({})\n'.format(str(timedelta(seconds=int(time()-t0)))))


def run_roi_extraction(path):
    register_path = r'{}\processed'.format(path)
    output_path = register_path

    # timer 
    from time import time
    from datetime import timedelta
    
    ops = np.load(r'{}\suite2p\plane0\ops.npy'.format(register_path), allow_pickle=True).item()
    # ops['input_format'] = 'binary'
    ops['sparse_mode'] = True
    ops['anatomical_only'] = False
    ops['roidetect'] = True
    ops['spatial_scale'] = 2
    ops['denoise'] = True
    
    # neuropil extraction parameters 
    ops['circular_neuropil'] = True
    ops['inner_neuropil_radius'] = 5
    
    # roi extraction parameters 
    ops['max_iterations'] = 1  # max_iterations=250 * ops["max_iterations"]
    ops['high_pass'] = 200
    
    ops['wang:bin_size'] = 1
    ops['wang:high_pass_overlapping'] = True
    ops['wang:rolling_width'] = 30
    ops["wang:rolling_bin"] = 'max'
    ops['wang:use_alt_norm'] = True
    ops['wang:downsample_scale'] = 1
    ops['wang:thresh_act_pix'] = 0.04
    ops['wang:thresh_peak_default'] = 0.03

    ops['wang:save_roi_iterations'] = True
    ops['save_path_new'] = output_path

    ops['wang:save_path_sparsedetect'] = output_path
    ops['wang:neuropil_lam']= True
    ops['wang:movie_chunk'] = 10000
    ops['wang:norm_method'] = 'max'

    # save output to 'run.log' file
    os.makedirs(output_path, exist_ok = True)
    log_path = output_path+r'\run_suite2p-wang-lab.log'
    print('running log saved to {}'.format(log_path))

    # run suite2p
    db = {
        'data_path': [str(path)],
        'save_path0': output_path,
        # 'save_path0': str(p_data)+r'\ROI_detection_test_2.0',
    }

    print('roi extraction starts...')
    t0 = time()
    with open(log_path, 'w') as f:
        with redirect_stdout(f):
            print('running suite2p v{} from Spyder'.format(suite2p.version))
            suite2p.run_s2p(ops=ops, db=db)

    print('roi extraction complete ({})\n'.format(str(timedelta(seconds=int(time()-t0)))))


#%% run all sessions
for path in pathGRABNE:
    sessname = path[-17:]
    print(sessname)
    
    reg_path = path+r'\processed'
    if not os.path.exists(reg_path):  # if registration has not been performed
        register(path)
    # if 'no tiffs' is raised, most likely it is due to typos in pathnames
    
    run_roi_extraction(path)
    
    
"""
note to self: put those ugly functions into another script please
"""