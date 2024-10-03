# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:47:21 2024

plots an example session with colour-coded single-trial traces

@author: Dinghao Luo
"""


#%% imports 
import sys
import scipy.io as sio 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from math import log 
from scipy.stats import poisson

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if (r'Z:\Dinghao\code_mpfi_dinghao\iutils' in sys.path) == False:
    sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf


#%% session 
pathname = r'Z:\Dinghao\MiceExp\ANMD071r\A071r-20230923\A071r-20230922-02'
recname = 'A071r-20230922-02'


#%% pre-post ratio-based classification 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 
pyr_act = []
for cluname, row in df.iterrows():
    if cluname.split(' ')[0]==recname:
        clu_ID = int(cluname.split(' ')[1][3:])
        if row['ctrl_pre_post']<=.8:
            pyr_act.append(clu_ID-2)  # HPCLC_all_train.py adds 2 to the ID, so we subtracts 2 here 
tot_pyract = len(pyr_act)


#%% main 
rasters = list(np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_all_rasters_npy_simp\{}.npy'.format(recname), 
                       allow_pickle=True).item().values())

# determine if each cell is pyramidal or intern 
info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
rec_info = info['rec'][0][0]
spike_rate = rec_info['firingRate'][0]

behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
stim_trials = np.where(stimOn!=0)[0]+1
tot_trial = len(stimOn)
tot_baseline = stim_trials[0]


# behaviour
lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(recname[1:5], recname[:14], recname[:17], recname[:17])
alignRun = sio.loadmat(lickfilename)

# -1 to account for MATLAB Python difference
delete = list(np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1)  # bad trials 

# ignore all 1st trials since it is before counting starts and is an empty cell
licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]

first_licks = []
for trial in range(tot_trial):
    lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
    if len(lk)!=0:  # append only if there is licks in this trial
        first_licks.append((lk[0]-starts[trial])/1250)
    else:
        first_licks.append(0)
first_licks_baseline = first_licks[:stim_trials[0]]

sorted_index = np.argsort(first_licks)
early = sorted_index[:20]; late = sorted_index[-20:]


#%% the Poisson deviation part 
tot_bins = 16
fig, axs = plt.subplots(1,2, figsize=(5.5,2.5)); xaxis = np.linspace(-2.5,5.5,tot_bins+1)[:-1]
pop_dev = []  # tot_trial-long
pol_pop_dev = []
for trial in early:
    pol_single_dev = np.zeros((tot_pyract, tot_bins))
    cell_counter = 0
    for clu in pyr_act:
        curr_raster = rasters[clu][trial]
        for tbin, t in enumerate(np.linspace(0,8,tot_bins+1)[:-1]):  # 3 seconds before, 5 seconds after 
            curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
            coeff=1
            if curr_bin<spike_rate[clu]: coeff=-1  # this is adding in polarity 
            pol_single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[clu]))*coeff
        cell_counter+=1
    axs[0].plot(xaxis, np.mean(pol_single_dev, axis=0), lw=1, alpha=.25, color='orchid')
for trial in late:
    pol_single_dev = np.zeros((tot_pyract, tot_bins))
    cell_counter = 0
    for clu in pyr_act:
        curr_raster = rasters[clu][trial]
        for tbin, t in enumerate(np.linspace(0,8,tot_bins+1)[:-1]):  # 3 seconds before, 5 seconds after 
            curr_bin = sum(curr_raster[int(t*1250):int((t+1)*1250)])
            coeff=1
            if curr_bin<spike_rate[clu]: coeff=-1  # this is adding in polarity 
            pol_single_dev[cell_counter, tbin] = -log(poisson.pmf(curr_bin, spike_rate[clu]))*coeff
        cell_counter+=1
    axs[0].plot(xaxis, np.mean(pol_single_dev, axis=0), lw=1, alpha=.25, color='darkred')