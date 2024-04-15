# -*- coding: utf-8 -*-
"""
Created by Nico Spiller
Modified and tested by Dinghao Luo 
"""

#%% imports 
import suite2p
import numpy as np

from pathlib import Path
from contextlib import redirect_stdout


#%% define list of folders
# option 1: manually
folders = [
    r'C:\dinghao_temp\A074-20231201-02',
]

# # option 2: glob expression
# folders = [ *Path(r'C:\temp').glob('A2*/') ]


#%%run suite2p
# default ops file
ops = np.load(r'Z:/Dinghao/MiceExp/axons_v2.0.npy', allow_pickle=True).item()

# optional: change parameters
# ops['spatial_scale'] = 2
# ops['two_step_registration'] = 1
ops['anatomical_only'] = 0


#%% cycle through folders
for pathname in folders:

    print('Running suite2p for {}'.format(pathname))
    
    # save output to 'run.log' file
    pathlog = pathname + 'run.log'    
    print('Saving text output to {}'.format(pathlog))

    # run suite2p
    db = {
        'data_path': [ str(pathname) ],
        'save_path0': str(pathname),
    }

    print('Running suite2p without GUI')
    suite2p.run_s2p(ops=ops, db=db)