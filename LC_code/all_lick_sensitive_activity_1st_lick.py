# -*- coding: utf-8 -*-
"""
Created on Mon 13 Nov 14:32:54 2023

Quantify the relationship between first lick time and neuronal activity drop-off for inhibition cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import scipy.io as sio
from scipy.stats import wilcoxon
import pandas as pd

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

mean_cutoff_all = []  # note that this, despite the name, does not include stims
mean_cutoff_stim = []  # this includes stims, not anything else

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
            curr_train = np.convolve(train[trial], gaus_spike, 'same')[tlick+3750-3750:tlick+3750+3750]
            shuf_train = cir_shuf(curr_train)
            
            # idea: get 1 mean shuffled value and use next() to find the crossing point
            mean_shuf = np.mean(shuf_train)
            
            try:
                co = next(k for k, value in enumerate(curr_train) if value < mean_shuf) + tlick
                cutoff.append(co)
                
            except:
                cutoff.append(100)  # if StopIteration, default to 100 to eliminate later
        
        else:  # if no licks in the current trial, default to 100 to eliminate later 
            cutoff.append(100)
        
    # remove trials where drop off happened before the window
    # and divide by 1250 to convert to seconds
    del_trial = []
    for trial, value in enumerate(cutoff):
        if value==first_licks[trial] or value==100:
            del_trial.append(trial)
    
    cutoff_all = [s/1250 for i, s in enumerate(cutoff) if i not in del_trial and i not in stim_trial]
    first_licks_all = [s/1250 for i, s in enumerate(first_licks) if i not in del_trial and i not in stim_trial]
    cutoff_stim = [s/1250 for i, s in enumerate(cutoff) if i not in del_trial and i in stim_trial]
    first_licks_stim = [s/1250 for i, s in enumerate(first_licks) if i not in del_trial and i in stim_trial]
    
    # plotting 
    fig, ax = plt.subplots(figsize=(2.6,2.6))
    fig.suptitle(cluname)
    ax.set(xlabel='time to inh. (s)', ylabel='time to 1st lick (s)',
           xlim=(1, 5), ylim=(1, 5),
           xticks=[2, 4], yticks=[2, 4])
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    
    ap = ax.scatter(cutoff_all, first_licks_all, s=8, ec='k', c='grey', linewidth=.75, alpha=.75)
    sp = ax.scatter(cutoff_stim, first_licks_stim, s=8, ec='royalblue', c='cornflowerblue', linewidth=.75)
    
    ax.legend([ap, sp], ['ctrl', 'stim'], frameon=False)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_inhibition_v_first_licks\{}.png'.format(cluname),
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

pval = wilcoxon(mean_cutoff_all_clean, mean_cutoff_stim_clean)[1]

fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([mean_cutoff_all_clean, mean_cutoff_stim_clean],
                   positions=[1, 2],
                   showextrema=False, showmedians=True)

colors = ['grey', 'royalblue']
for i in range(2):
    vp['bodies'][i].set_color(colors[i])
    vp['bodies'][i].set_edgecolor('k')
vp['cmedians'].set(color='darkred', lw=2)

ax.scatter([1]*len(mean_cutoff_all_clean), 
           mean_cutoff_all_clean, 
           s=3, c='grey', ec='none', alpha=.5)

ax.scatter([2]*len(mean_cutoff_stim_clean), 
           mean_cutoff_stim_clean, 
           s=3, c='royalblue', ec='none', alpha=.5)

ax.plot([[1]*len(mean_cutoff_all_clean), [2]*len(mean_cutoff_stim_clean)], [mean_cutoff_all_clean, mean_cutoff_stim_clean], 
        color='grey', alpha=.25, linewidth=1)
ax.plot([1, 2], [np.median(mean_cutoff_all_clean), np.median(mean_cutoff_stim_clean)],
        color='darkred', linewidth=1.5)

ax.set(xlim=(.5,2.5), ylim=(2,5.2),
       yticks=[2, 3, 4, 5],
       ylabel='time to inh. (s)',
       title='time to inh.\np={}'.format(np.round(pval, 8)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl', 'stim'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_ttinhibition.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_020_ttinhibition.pdf',
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\paper\figures\figure_2_opto_time_to_inhibition.pdf',
            bbox_inches='tight')

plt.close(fig)