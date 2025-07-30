# -*- coding: utf-8 -*-
"""
Created on 4 June 16:30:51 2024
Modified on 8 July 2025

**NOTE: THIS SCRIPT RUNS THE CUSTOMISED SUITE2P-WANG-LAB INSTEAD OF SUITE2P**
**SUITE2P-WANG-LAB CAN BE ACCESSED HERE: 
    https://github.com/the-wang-lab/suite2p-wang-lab**
    
Modified to work on any directory list and as a general-purpose batch registra-
    tion processor 

@author: Dinghao Luo
"""


#%% imports 
import sys
import os

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\imaging_code\utils')
import suite2p_functions as s2f

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list

which = input('''
    Process which experiment?
    1. GRABNE,
    2. HPCdLightLCOpto,
    3. HPCdLightLCOptoCtrl,
    4. HPCdLightLCOptoInh, 
    5. PROCESS ALL\n
    ''')

if which == '1':
    paths = rec_list.pathHPCGRABNE
    prefix = 'HPCGRABNE'
elif which == '2':
    paths = rec_list.pathdLightLCOpto
    prefix = 'HPCdLightLCOpto'
elif which == '3':
    paths = rec_list.pathdLightLCOptoCtrl
    prefix = 'HPCdLightLCOptoCtrl'
elif which == '4':
    paths = rec_list.pathdLightLCOptoInh
    prefix = 'HPCdLightLCOptoInh'
elif which == '5':
    paths = (
        rec_list.pathHPCGRABNE + 
        rec_list.pathdLightLCOpto +
        rec_list.pathdLightLCOptoCtrl + 
        rec_list.pathdLightLCOptoInh
        )
    

#%% run all sessions
for path in paths:
    sessname = path[-17:]
    print('\n{}'.format(sessname))
    
    reg_path = os.path.join(path, 'suite2p')
    reg_path_alt = os.path.join(path, 'processed')  # in some older recordings
    if os.path.exists(reg_path) or os.path.exists(reg_path_alt):  # if registed
        print('session already registered; skip')
        continue
    else:
        s2f.register(path)
        
    # if 'no tiffs' is raised, most likely it is due to typos in pathnames