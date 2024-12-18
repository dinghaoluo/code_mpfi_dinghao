# -*- coding: utf-8 -*-
"""
Created on Mon 13 Nov 14:32:54 2023

Quantify the relationship between first lick time and neuronal activity drop-off for inhibition cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
import os
import sys

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load dataframe  
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% load trains 
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()


#%% shuffle function 
def cir_shuf(train, num_shuf=50):
    tot_t = len(train)
    shuf_array = np.zeros([num_shuf, tot_t])
    for i in range(num_shuf):
        rand_shift = np.random.randint(1, tot_t)
        shuf_array[i,:] = np.roll(train, -rand_shift)
    
    return np.percentile(shuf_array, 75, axis=0)  # upper quartile
    

def find_switch_point(line, v, mode='downwards'):
    # returns the crossing point; mode='upwards' or 'downwards'
    if mode=='downwards':
        for i in range(1, len(line)):
            if line[i - 1] >= v and line[i] < v:
                return i
    elif mode=='upwards':
        for i in range(1, len(line)):
            if line[i - 1] <= v[i - 1] and line[i] > v[i]:
                return i
            
    return 10000


#%% find cells
clu_list = list(cell_prop.index)

sensitive = []
exc = []; inh = []

for clu in cell_prop.index:
    sens = cell_prop['lick_sensitive'][clu]
    stype = cell_prop['lick_sensitive_type'][clu]
    
    if sens:
        sensitive.append(clu)
        if stype=='excitation':
            exc.append(clu)
        if stype=='inhibition':
            inh.append(clu)
            
            
#%% main (inhibition)
gx_spike = np.arange(-500, 500, 1)
sigma_spike = 1250/3
gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]

xaxis_trial = np.arange(1250*6)/1250-3

mean_cutoff_all = []  # note that this, despite the name, does not include stims
mean_cutoff_stim = []  # this includes stims and nothing else

for cluname in inh:
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
                outdir = r'Z:\Dinghao\code_dinghao\LC_all\lick_sensitive_cutoff_point_analysis\{}\trial{}.png'.format(cluname+' OFF', trial+1)
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
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_inhibition_v_first_licks\{}.png'.format(cluname+' OFF'),
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
                            'grey', 'royalblue', 
                            paired=True, 
                            xticklabels=['ctrl.', 'stim'], 
                            ylabel='t. to OFF', 
                            title='1st-lick OFF', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_time_to_OFF', dpi=300)


#%% main (excitation)
gx_spike = np.arange(-500, 500, 1)
sigma_spike = 1250/3
gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]

xaxis_trial = np.arange(1250*6)/1250-3

mean_cutoff_all = []  # note that this, despite the name, does not include stims
mean_cutoff_stim = []  # this includes stims and nothing else

for cluname in exc:
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
            curr_train = np.convolve(train[trial], gaus_spike, 'same')[tlick+3750-3750:tlick+3750+3750]*1250
            shuf_train = cir_shuf(curr_train)
            
            # idea: get 1 mean shuffled value and use next() to find the crossing point
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
                outdir = r'Z:\Dinghao\code_dinghao\LC_all\lick_sensitive_cutoff_point_analysis\{}\trial{}.png'.format(cluname+' ON', trial+1)
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
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_inhibition_v_first_licks\{}.png'.format(cluname+' ON'),
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
                            'grey', 'royalblue', 
                            paired=True, 
                            xticklabels=['ctrl.', 'stim'], 
                            ylabel='t. to ON', 
                            title='1st-lick ON', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_time_to_ON', dpi=300)