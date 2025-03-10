# -*- coding: utf-8 -*-
"""
Created on Fri 7 Mar 16:05:21 2025

plot ISIs of single neurones

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import sys 

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% plot single-cell ISIs
for path in paths:
    recname = path[-17:]
    print(f'plotting {recname}')
    
    sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
    
    ISI_dict = np.load(
        rf'{sess_folder}\{recname}_all_ISIs.npy',
        allow_pickle=True
        ).item()
    
    identity_dict = np.load(
        rf'{sess_folder}\{recname}_all_identities.npy',
        allow_pickle=True
        ).item()
    
    for clu, ISIs in ISI_dict.items():
        
        fig, ax = plt.subplots(figsize=(3,2))
        
        cluname = f'{clu} tagged' if identity_dict[clu] else clu
        
        ax.hist([ISI/20000 for ISI in ISIs], bins=np.arange(0, 2, .01),
                color='k')
        
        ax.set(title=cluname,
               xlabel='ISI (s)',
               ylabel='frequency')
        
        fig.savefig(
            rf'Z:\Dinghao\code_dinghao\LC_ephys\single_cell_ISIs\{cluname}.png',
            dpi=300,
            bbox_inches='tight')
        
        plt.close(fig)