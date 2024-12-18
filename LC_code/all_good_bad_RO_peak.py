# -*- coding: utf-8 -*-
"""
Created on Sun June 11 13:12:54 2023
modified on 12 Dec 2024 to tidy up analysis

LC: visual and statistical comparison between good and bad trial RO peaks 

*use tagged + putative Dbh RO peaking cells*

***UPDATED GOOD/BAD TRIALS***
bad trial parameters 12 Dec 2024 (in the .pkl dataframe):
    rewarded == -1
    noFullStop
    licks before 90

@author: Dinghao Luo
"""


#%% imports
import os 
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt 
import scipy.io as sio

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf
from common import mpl_formatting 
mpl_formatting()

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% GPU acceleration
try:
    import cupy as cp 
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0  # check if an NVIDIA GPU is available
except ModuleNotFoundError:
    print('CuPy is not installed; see https://docs.cupy.dev/en/stable/install.html for installation instructions')
    GPU_AVAILABLE = False
except Exception as e:
    # catch any other unexpected errors and print a general message
    print(f'An error occurred: {e}')
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    # we are assuming that there is only 1 GPU device
    xp = cp
    from coommon import sem_gpu as sem
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('UTF-8')
    print(f'GPU-acceleration with {str(name)} and cupy')
else:
    import numpy as np 
    from scipy.stats import sem 
    xp = np
    print('GPU-acceleartion unavailable')


#%% functions 
def load_beh_series(df_filename, recname):
    return pd.read_pickle(df_filename).loc[recname]

def find_RO_peaking(cell_prop):
    RO_peaking_keys = []
    for cell in cell_prop.index:
        pk = cell_prop['peakness'][cell]  # union-peakness
        pt = cell_prop['putative'][cell]  # putative
        tg = cell_prop['tagged'][cell]
        
        if pk and (pt or tg):
            RO_peaking_keys.append(cell)

    return RO_peaking_keys

def get_good_bad_idx(beh_series):
    bad_trial_map = beh_series['bad_trials']
    good_idx = [trial for trial, quality in enumerate(bad_trial_map) if not quality]
    bad_idx = [trial for trial, quality in enumerate(bad_trial_map) if quality]
    
    return good_idx, bad_idx

def get_trialtype_idx(beh_filename):
    behPar = sio.loadmat(beh_filename)
    stim_idx = xp.where(behPar['behPar']['stimOn'][0][0][0]!=0)[0]
    
    if len(stim_idx)>0:
        return xp.arange(1, stim_idx[0]), stim_idx, stim_idx+2  # stim_idx+2 are indices of control trials
    else:
        return np.arange(1, len(behPar['behPar']['stimOn'][0][0][0])), [], []  # if no stim trials


#%% load data 
trains = np.load(r'Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                 allow_pickle=True).item()
cell_prop = pd.read_pickle(r'Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% MAIN
# get RO-peaking cells 
RO_peaking = find_RO_peaking(cell_prop)

pk_good = []; pk_bad = []
s2n_good = []; s2n_bad = []
p_good = {}; p_good_sem = {}  # single trials
p_bad = {}; p_bad_sem = {}  # single trials
max_length = 12500  # max length for trial analysis

for pathname in pathLC:
    recname = pathname[-17:]
    print(recname)
    
    # load behaviour
    beh_df = load_beh_series(r'Z:\Dinghao\code_dinghao\behaviour\all_LC_sessions.pkl', recname)
    
    # import bad beh trial indices
    behPar = sio.loadmat(pathname+pathname[-18:]+
                         '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
    # -1 to account for MATLAB Python difference
    bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
    # -1 to account for 0 being an empty trial
    good_idx = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    good_idx = np.delete(good_idx, bad_idx)
    
    # import stim trial indices
    baseline_idx, _, _ = get_trialtype_idx(
        r'{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.
        format(pathname, recname)
        )
    
    # import tagged cell spike trains from all_tagged_train
    if len(bad_idx) >= 10:  # 10 bad trials at least, prevents contam.
        for name in RO_peaking:
            if name[:17] == recname:
                curr = trains[name]  # train of current clu
                curr_good = np.zeros([len(good_idx), max_length])
                curr_bad = np.zeros([len(bad_idx), max_length])
                for i, trial in enumerate(good_idx):
                    curr_length = len(curr[trial])
                    curr_good[i, :curr_length] = curr[trial][:max_length]
                for i, trial in enumerate(bad_idx):
                    curr_length = len(curr[trial])
                    curr_bad[i, :curr_length] = curr[trial][:max_length]
                p_good[name] = xp.mean(curr_good, axis=0)
                p_good_sem[name] = sem(curr_good, axis=0)
                pk_good.append(xp.mean(p_good[name][3125:4375]))
                bl_good = (xp.mean(p_good[name][625:1875])+xp.mean(p_good[name][5625:6875]))/2
                s2n_good.append(xp.mean(p_good[name][3125:4375])/bl_good)
                p_bad[name] = xp.mean(curr_bad, axis=0)
                p_bad_sem[name] = sem(curr_bad, axis=0)
                bl_bad = (xp.mean(p_bad[name][625:1875])+xp.mean(p_bad[name][5625:6875]))/2
                s2n_bad.append(xp.mean(p_bad[name][3125:4375])/bl_bad)
                pk_bad.append(xp.mean(p_bad[name][3125:4375]))


#%% calculation 
p_g_avg = []
p_b_avg = []
for clu in list(p_good.items()):
    p_g_avg.append(clu[1])
for clu in list(p_bad.items()):
    p_b_avg.append(clu[1])
p_g_sem = sem(p_g_avg, axis=0)
p_g_avg = np.mean(p_g_avg, axis=0)
p_b_sem = sem(p_b_avg, axis=0)
p_b_avg = np.mean(p_b_avg, axis=0)


#%% plotting
print('\nplotting avg onset-bursting good vs bad spike trains...')
tot_plots = len(p_good)  # total number of cells
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()
xaxis = np.arange(-3*1250, 7*1250, 1)/1250 

for i in range(tot_plots):
    curr_clu_good = list(p_good.items())[i]
    curr_clu_name = curr_clu_good[0]
    curr_good_avg = curr_clu_good[1]
    curr_good_sem = p_good_sem[curr_clu_name]
    curr_bad_avg = p_bad[curr_clu_name]
    curr_bad_sem = p_bad[curr_clu_name]
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(ylim=(0, np.max(curr_good_avg)*1250*1.5),
           ylabel='spike rate (Hz)',
           xlabel='time (s)')
    good_avg = ax.plot(xaxis, curr_good_avg*1250, color='seagreen')
    good_sem = ax.fill_between(xaxis, curr_good_avg*1250+curr_good_sem*1250,
                                      curr_good_avg*1250-curr_good_sem*1250,
                                      color='springgreen')
    bad_avg = ax.plot(xaxis, curr_bad_avg*1250, color='firebrick', alpha=.3)
    bad_sem = ax.fill_between(xaxis, curr_bad_avg+curr_bad_sem,
                                     curr_bad_avg-curr_bad_sem,
                                     color='lightcoral')
    ax.vlines(0, 0, 20, color='grey', alpha=.25)

plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)
fig.savefig(out_directory + '\\'+'LC_all_goodvbad_(alignedRun)_putDbh_ROpeaking.png')


#%% avg profile for onset-bursting clus
print('\nplotting avg onset-bursting good vs bad averaged spike trains...')

pval_wil = wilcoxon(pk_good, pk_bad)[1]

xaxis = np.arange(-3*1250, 7*1250, 1)/1250 

p_g_burst_avg = []
p_b_burst_avg = []
for clu in list(p_good.items()):
    p_g_burst_avg.append(clu[1])
    p_b_burst_avg.append(p_bad[clu[0]])
p_g_burst_sem = sem(p_g_burst_avg, axis=0)
p_g_burst_avg = np.mean(p_g_burst_avg, axis=0)
p_b_burst_sem = sem(p_b_burst_avg, axis=0)
p_b_burst_avg = np.mean(p_b_burst_avg, axis=0)

fig, ax = plt.subplots(figsize=(2,1.4))
p_good_ln, = ax.plot(xaxis, p_g_burst_avg*1250, color='forestgreen')
p_bad_ln, = ax.plot(xaxis, p_b_burst_avg*1250, color='grey')
ax.fill_between(xaxis, p_g_burst_avg*1250+p_g_burst_sem*1250, 
                       p_g_burst_avg*1250-p_g_burst_sem*1250,
                       color='mediumseagreen', alpha=.1, edgecolor='none')
ax.fill_between(xaxis, p_b_burst_avg*1250+p_b_burst_sem*1250, 
                       p_b_burst_avg*1250-p_b_burst_sem*1250,
                       color='gainsboro', alpha=.1, edgecolor='none')
# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(title='good v bad trials (all Dbh+)',
       ylim=(1.8,9.6),yticks=[2,5,8],
       xlim=(-1,4),xticks=[0,2,4],
       ylabel='spike rate (Hz)',
       xlabel='time from run-onset (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([p_good_ln, p_bad_ln], 
          ['good trial', 'bad trial'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [9,9], c='k', lw=.5)
plt.text(0, 9, 'pval_wil={}'.format(round(pval_wil, 5)), ha='center', va='bottom', color='k', fontsize=5)

for ext in ['png', 'pdf']:
    fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_goodvbad_(alignedRun)_avg_ROpeaking.{}'.format(ext),
                dpi=300,
                bbox_inches='tight')


#%% statistics 
pf.plot_violin_with_scatter(s2n_good, s2n_bad, 
                            'darkgreen', 'grey', 
                            paired=True, 
                            xticklabels=['good\ntrials', 'bad\ntrials'], 
                            ylabel='run-onset burst SNR', 
                            title='LC RO-SNR', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_all\LC_all_goodvbad_(alignedRun)_avg_ROpeaking_violin', 
                            dpi=300)