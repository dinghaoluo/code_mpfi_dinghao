# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:53:55 2025

LC recording processing pipeline

**note**:
    this pipeline assumes that the MATLAB preprocessing has been run; see 
    '~\code_mpfi_dinghao\matlab_preprocessing\' for run_all scripts

@author: Dinghao Luo
"""


#%% import scripts
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\LC_code')
import LC_all_spikes_ISIs, LC_all_waveforms_acgs, LC_all_extract, LC_all_identity_UMAP, LC_all_profiles


#%% pipeline proper 

'''
1) get the spike times and ISIs of each recorded LC unit
'''
LC_all_spikes_ISIs.main()

'''
2) get the waveforms, acgs and tagged vs non-tagged (identities) of recorded 
    LC units and place them into recording folders
    i.e. ..._all_identities.npy; ..._all_waveforms.npy; ..._all_ACGs.npy
'''
LC_all_waveforms_acgs.main()

'''
3) extract spike rasters (spike maps, 0's and 1's) and spike trains (smoothed)
    and place them into recording folders 
    i.e. ..._all_rasters.npy; ..._all_trains.npy
'''
LC_all_extract.main()

'''
4) apply UMAP-k-means clustering onto spike ACGs of recorded units, and then 
    using ACGs of tagged units as references, classify units as either putative
    Dbh+ or putative Dbh-. K-means results are saved as a dictionary and placed
    outside of recording folders, in the main result folder (...UMAP\)
'''
LC_all_identity_UMAP.main()

'''
5) finally, summarise unit properties and place them into a pandas dataframe 
    which is then saved into the main result folder
'''
LC_all_profiles.main()
