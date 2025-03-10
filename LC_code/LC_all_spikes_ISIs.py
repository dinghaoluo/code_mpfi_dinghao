# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:45:04 2025

calculate ISIs of single neurones

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import sys
import os

sys.path.append(r'Z:\Dinghao\code_dinghao')
import rec_list
paths = rec_list.pathLC

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
from param_to_array import param2array
mpl_formatting()


#%% main 
def main():
    for pathname in paths:
        recname = pathname[-17:]
        print('\n\nProcessing {}'.format(recname))
        
        sess_folder = rf'Z:\Dinghao\code_dinghao\LC_ephys\all_sessions\{recname}'
        os.makedirs(sess_folder, exist_ok=True)
        
        clu = param2array(rf'{pathname}/{recname}.clu.1')  # load .clu
        res = param2array(rf'{pathname}/{recname}.res.1')  # load .res
        
        all_clu = [int(c) for c in np.unique(clu)
                   if c not in ('', '0', '1') and 
                   len(np.where(clu==c)[0]) != 1]
        tot_clu = len(all_clu)
        
        spike_dict = {
            f'{recname} clu{clu_idx}': [
                int(t) 
                for i, t in enumerate(res[:-1])  # last element is ''
                if int(clu[i])==clu_idx]
            for clu_idx in np.arange(2, tot_clu+2)
            }
        
        ISI_dict = {
            key: np.diff(spikes)
            for key, spikes in spike_dict.items()
            }
        
        np.save(rf'{sess_folder}\{recname}_all_spikes.npy',
                spike_dict)
        np.save(rf'{sess_folder}\{recname}_all_ISIs.npy',
                ISI_dict)
        
if __name__ == '__main__':
    main()