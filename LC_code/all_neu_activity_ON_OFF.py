# -*- coding: utf-8 -*-
"""
Created on Sat 19 Oct 14:31:52 2024

Quantify the responses of neuronal activity changes against 1st-lick time
shuffle-based

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
import os
import sys

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if CUDA GPU is available
except ImportError:
    GPU_AVAILABLE = False  # CuPy not installed or no GPU

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device (we are poor)
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print('GPU detected: {}\nusing CuPy to accelerate computation'.format(str(name)))
else:
    name = 'NA'
    print('no GPU detected; falling back to NumPy')

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf
from common import gaussian_kernel_unity, mpl_formatting
mpl_formatting()


#%% params
single_trial_num_shuf = 500
gx_spike = np.arange(-500, 500, 1)
sigma_spike = 1250/3

gaus_spike = gaussian_kernel_unity(sigma_spike, GPU_AVAILABLE)

xaxis_trial = np.arange(1250*6)/1250-3


#%% timer
from time import time 
start = time()


#%% load dataframe  
cell_prop = pd.read_pickle(r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_single_cell_properties.pkl')


#%% load trains 
all_train = np.load('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_info.npy',
                    allow_pickle=True).item()
print('all spike trains loaded')


#%% shuffle function 
def cir_shuf(train, num_shuf=single_trial_num_shuf):
    tot_t = len(train)
    if GPU_AVAILABLE: 
        train_gpu = cp.array(train)
        shifts = cp.random.randint(1, tot_t, size=num_shuf)
        shuf_array = cp.array([cp.roll(train_gpu, -shift) for shift in shifts])
        return shuf_array.get()  # convert back to np array
    else:
        shifts = np.random.randint(1, tot_t, size=num_shuf)
        shuf_array = np.array([np.roll(train, -shift) for shift in shifts])
        return shuf_array
    

def find_switch_point(line, v, mode='downwards'):
    # returns the crossing point; mode='upwards' or 'downwards'
    if mode=='downwards':
        for i in range(1, len(line)):
            if line[i - 1] >= v and line[i] < v:
                return i
    elif mode=='upwards':
        for i in range(1, len(line)):
            if line[i - 1] <= v and line[i] > v:
                return i
    return 10000


#%% find cells
clu_list = list(cell_prop.index)

sensitive = []
ON = []; OFF = []

for clu in cell_prop.index:
    stype = cell_prop['lick_sensitive_shuf_type'][clu]
    
    if stype=='ON':
        ON.append(clu)
    if stype=='OFF':
        OFF.append(clu)
            
            
#%% main (OFF)
mean_cutoff_all = []  # note that this, despite the name, does not include stims
mean_cutoff_stim = []  # this includes stims and nothing else

for cluname in OFF:
    print(cluname)
    
    train = all_train[cluname]  # read all trials 
    tot_trial = train.shape[0]-1
    
    # get 1st lick time of each trial
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    
    # get stim trials
    behInfo = sio.loadmat('Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17]))['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1  # -1 to match up with matlab indexing
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])
    
    cutoff = []
    for trial in range(tot_trial):        
        tlick = first_licks[trial]
        
        if tlick<10000:  # only execute if there is actually licks in this trial
            if GPU_AVAILABLE:
                train_gpu = cp.array(train[trial])
                curr_train_gpu = cp.convolve(train_gpu, gaus_spike, mode='same')[tlick+3750-3750:tlick+3750+3750]*1250
                curr_train = curr_train_gpu.get()
            else:
                curr_train = np.convolve(train[trial], gaus_spike, 'same')[tlick+3750-3750:tlick+3750+3750]*1250
            shuf_train = cir_shuf(curr_train)
            
            # idea: get 1 mean shuffled value and use next() to find the crossing point
            mean_shuf = np.mean(shuf_train)
            
            try:
                switch_point = find_switch_point(curr_train[2500:5000], mean_shuf, mode='downwards')
                if switch_point==10000:
                    co = 10000
                else:
                    co = switch_point + tlick - 1250
                cutoff.append(co)
                
                # single-trial plots for visual inspection, 14 Oct 2024 Dinghao 
                fig, ax = plt.subplots(figsize=(2,1.3))
                ax.plot(xaxis_trial[:len(curr_train)], curr_train)
                ax.axhspan(mean_shuf+.001, mean_shuf-.001, color='grey', alpha=.5)
                ax.axvspan((co-tlick-1)/1250, (co-tlick+1)/1250, color='r')
                ax.set(title='trial {}'.format(trial+1))
                outdir = r'Z:\Dinghao\code_dinghao\LC_ephys\lick_sensitive_cutoff_point_analysis_shufbased\{}\trial{}.png'.format(cluname+' OFF', trial+1)
                os.makedirs(os.path.dirname(outdir), exist_ok=True)
                fig.savefig(outdir,
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                
            except StopIteration:
                cutoff.append(10000)  # if StopIteration, default to 100 to eliminate later
        
        else:  # if no licks in the current trial, default to 100 to eliminate later 
            cutoff.append(10000)
        
    # remove trials where drop off happened before the window
    # and divide by 1250 to convert to seconds
    del_trial = []
    for trial, value in enumerate(cutoff):
        if value==first_licks[trial] or value==10000:
            del_trial.append(trial)
    
    cutoff_all = [s/1250 for i, s in enumerate(cutoff) if i not in del_trial and i not in stim_trial]
    first_licks_all = [s/1250 for i, s in enumerate(first_licks) if i not in del_trial and i not in stim_trial]
    cutoff_stim = [s/1250 for i, s in enumerate(cutoff) if i not in del_trial and i in stim_trial]
    first_licks_stim = [s/1250 for i, s in enumerate(first_licks) if i not in del_trial and i in stim_trial]
    
    # plotting 
    fig, ax = plt.subplots(figsize=(2.6,2.6))
    fig.suptitle(cluname)
    ax.set(xlabel='time to OFF (s)', ylabel='time to 1st lick (s)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ap = ax.scatter(cutoff_all, first_licks_all, s=8, ec='k', c='grey', linewidth=.75, alpha=.75)
    sp = ax.scatter(cutoff_stim, first_licks_stim, s=8, ec='royalblue', c='cornflowerblue', linewidth=.75)
    
    ax.legend([ap, sp], ['ctrl', 'stim'], frameon=False)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_ephys\single_cell_OFF_v_first_licks_shufbased\{}.png'.format(cluname+' OFF'),
                dpi=500,
                bbox_inches='tight')
    
    plt.close(fig)
    
    
    # save to lists for statistics 
    mean_cutoff_all.append(np.median(cutoff_all))
    mean_cutoff_stim.append(np.median(cutoff_stim))  # use median to weaken outliers
    

#%% statistics and plotting 
# first eliminate sessions without stimulations
del_sess = []
for i, val in enumerate(mean_cutoff_stim):
    if np.isnan(val)==True:
        del_sess.append(i)
mean_cutoff_all_clean = [s for i, s in enumerate(mean_cutoff_all) if i not in del_sess]
mean_cutoff_stim_clean = [s for i, s in enumerate(mean_cutoff_stim) if i not in del_sess]

pf.plot_violin_with_scatter(mean_cutoff_all_clean, mean_cutoff_stim_clean, 
                            'grey', 'forestgreen', 
                            paired=True, 
                            xticklabels=['ctrl.', 'stim'], 
                            ylabel='t. to OFF', 
                            title='1st-lick OFF', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_time_to_OFF_shufbased', dpi=300)


#%% main (ON)
mean_cutoff_all = []  # note that this, despite the name, does not include stims
mean_cutoff_stim = []  # this includes stims and nothing else

for cluname in ON:
    print(cluname)
    
    train = all_train[cluname]  # read all trials 
    tot_trial = train.shape[0]-1
    
    # get 1st lick time of each trial
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    
    # get stim trials
    behInfo = sio.loadmat('Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17]))['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1  # -1 to match up with matlab indexing
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])
    
    cutoff = []
    for trial in range(tot_trial):        
        tlick = first_licks[trial]
        
        if tlick<10000:  # only execute if there is actually licks in this trial
            if GPU_AVAILABLE:
                train_gpu = cp.array(train[trial])
                curr_train_gpu = cp.convolve(train_gpu, gaus_spike, mode='same')[tlick+3750-3750:tlick+3750+3750]*1250
                curr_train = curr_train_gpu.get()
            else:
                curr_train = np.convolve(train[trial], gaus_spike, 'same')[tlick+3750-3750:tlick+3750+3750]*1250
            shuf_train = cir_shuf(curr_train)
            
            mean_shuf = np.mean(shuf_train)
            
            try:
                switch_point = find_switch_point(curr_train[2500:5000], mean_shuf, mode='upwards')
                if switch_point==10000:
                    co = 10000
                else:
                    co = switch_point + tlick - 1250
                cutoff.append(co)
                
                # single-trial plots for visual inspection, 14 Oct 2024 Dinghao 
                fig, ax = plt.subplots(figsize=(2,1.3))
                ax.plot(xaxis_trial[:len(curr_train)], curr_train)
                ax.axhspan(mean_shuf+.001, mean_shuf-.001, color='grey', alpha=.5)
                ax.axvspan((co-tlick-1)/1250, (co-tlick+1)/1250, color='r')
                ax.set(title='trial {}'.format(trial+1))
                outdir = r'Z:\Dinghao\code_dinghao\LC_ephys\lick_sensitive_cutoff_point_analysis_shufbased\{}\trial{}.png'.format(cluname+' ON', trial+1)
                os.makedirs(os.path.dirname(outdir), exist_ok=True)
                fig.savefig(outdir,
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                
            except StopIteration:
                cutoff.append(10000)  # if StopIteration, default to 100 to eliminate later
        
        else:  # if no licks in the current trial, default to 100 to eliminate later 
            cutoff.append(10000)
        
    # remove trials where drop off happened before the window
    # and divide by 1250 to convert to seconds
    del_trial = []
    for trial, value in enumerate(cutoff):
        if value==first_licks[trial] or value==10000:
            del_trial.append(trial)
    
    cutoff_all = [s/1250 for i, s in enumerate(cutoff) if i not in del_trial and i not in stim_trial]
    first_licks_all = [s/1250 for i, s in enumerate(first_licks) if i not in del_trial and i not in stim_trial]
    cutoff_stim = [s/1250 for i, s in enumerate(cutoff) if i not in del_trial and i in stim_trial]
    first_licks_stim = [s/1250 for i, s in enumerate(first_licks) if i not in del_trial and i in stim_trial]
    
    # plotting 
    fig, ax = plt.subplots(figsize=(2.6,2.6))
    fig.suptitle(cluname)
    ax.set(xlabel='time to ON (s)', ylabel='time to 1st lick (s)',
           xlim=(1, 5), ylim=(1, 5),
           xticks=[2, 4], yticks=[2, 4])
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ap = ax.scatter(cutoff_all, first_licks_all, s=8, ec='k', c='grey', linewidth=.75, alpha=.75)
    sp = ax.scatter(cutoff_stim, first_licks_stim, s=8, ec='royalblue', c='cornflowerblue', linewidth=.75)
    
    ax.legend([ap, sp], ['ctrl', 'stim'], frameon=False)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_ephys\single_cell_ON_v_first_licks_shufbased\{}.png'.format(cluname+' ON'),
                dpi=500,
                bbox_inches='tight')
    
    plt.close(fig)
    
    
    # save to lists for statistics 
    mean_cutoff_all.append(np.median(cutoff_all))
    mean_cutoff_stim.append(np.median(cutoff_stim))  # use median to weaken outliers
    

#%% statistics and plotting 
# first eliminate sessions without stimulations
del_sess = []
for i, val in enumerate(mean_cutoff_stim):
    if np.isnan(val)==True:
        del_sess.append(i)
mean_cutoff_all_clean = [s for i, s in enumerate(mean_cutoff_all) if i not in del_sess]
mean_cutoff_stim_clean = [s for i, s in enumerate(mean_cutoff_stim) if i not in del_sess]

pf.plot_violin_with_scatter(mean_cutoff_all_clean, mean_cutoff_stim_clean, 
                            'grey', 'darkred', 
                            paired=True, 
                            xticklabels=['ctrl.', 'stim'], 
                            ylabel='t. to ON', 
                            title='1st-lick ON', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_time_to_ON_shufbased', dpi=300)