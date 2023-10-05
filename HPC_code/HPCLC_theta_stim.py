# -*- coding: utf-8 -*-
"""
Created on Sat 5 Aug 14:23:46 2023

theta analysis aligned to LC stim

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import mat73
import sys 
from scipy.stats import circvar, wilcoxon, sem
from math import log10

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% functions
def conv_db(x):
    if isinstance(x, np.ndarray):
        return [20*log10(i) for i in x]
    else:
        return 20*log10(x)


#%% plotting parameters 
taxis = np.arange(-1250, 6250)/1250
plt.rcParams['font.family'] = 'Arial' 


#%% main 
all_stim_theta_amplitude = []
all_start_theta_amplitude = []; all_cont_theta_amplitude = []; all_recov_theta_amplitude = []

all_stim_cir_dev = []
all_start_cir_dev = []; all_cont_cir_dev = []; all_recov_cir_dev = []

for sessname in pathHPC:
    recname = sessname[43:60]  # recording name, e.g. A069r-20230905-01
    theta = mat73.loadmat('{}/{}_eeg_1250Hz.mat'.format(sessname, recname))

    try: 
        theta_h = theta['ThetaPhase_hilbert'][:,0]
        theta_amp = theta['ThetaAmp_hilbert'][:,0]
    except IndexError:  # when there is only one shank
        theta_h = theta['ThetaPhase_hilbert'][:]
        theta_amp = theta['ThetaAmp_hilbert'][:]

    mat_BTDT = sio.loadmat('{}/{}BTDT.mat'.format(sessname, recname))
    behEvents = mat_BTDT['behEventsTdt']
    stim_tps = behEvents['stimPulse'][0,0][:,1]

    stims = []
    last_stim = stim_tps[0]; stims.append(last_stim)
    for t in stim_tps[1:]:
        if t-last_stim < 1250:  # 1 second within the last stim 
            pass
        else:
            last_stim = t
            stims.append(t)
        
    alignRun = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname, recname))
    # alignCue = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat'.format(sessname, recname))
    behInfo = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname, recname))['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))
    stim_cont = stim_trial+2
    tot_trial = len(behInfo['pulseMethod'][0][0][0])
        
    # cues = alignCue['trialsCue']['startLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]


    # data wrangling
    peri_stim_theta_amp = []
    peri_stim_theta_phase = []
    for t in stim_trial:
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # -1~5 s 
        peri_stim_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_stim_theta_phase.append(phase_seq)
    avg_peri_stim_theta_amplitude = np.mean(peri_stim_theta_amp, axis=0)
    all_stim_theta_amplitude.append(avg_peri_stim_theta_amplitude)
    
    peri_stim_cir_dev = circvar(peri_stim_theta_phase, high=3.14159, low =-3.14159,
                                axis=0)
    all_stim_cir_dev.append(peri_stim_cir_dev)
    
    
    peri_start_theta_amp = []
    peri_start_theta_phase = []
    for t in np.arange(stim_trial[0]):
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # 3 s around 
        peri_start_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_start_theta_phase.append(phase_seq)
    avg_peri_start_theta_amplitude = np.mean(peri_start_theta_amp, axis=0)
    all_start_theta_amplitude.append(avg_peri_start_theta_amplitude)
        
    peri_start_cir_dev = circvar(peri_start_theta_phase, high=3.14159, low =-3.14159,
                                 axis=0)
    all_start_cir_dev.append(peri_start_cir_dev)
    
    
    peri_cont_theta_amp = []
    peri_cont_theta_phase = []
    for t in stim_cont:
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # 3 s around 
        peri_cont_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_cont_theta_phase.append(phase_seq)
    avg_peri_cont_theta_amplitude = np.mean(peri_cont_theta_amp, axis=0)
    all_cont_theta_amplitude.append(avg_peri_cont_theta_amplitude)
        
    peri_cont_cir_dev = circvar(peri_cont_theta_phase, high=3.14159, low =-3.14159,
                                axis=0)
    all_cont_cir_dev.append(peri_cont_cir_dev)
    
    
    peri_recov_theta_amp = []
    peri_recov_theta_phase = []
    for t in np.arange(stim_trial[-1],tot_trial-1):
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # 3 s around 
        peri_recov_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_recov_theta_phase.append(phase_seq)
    avg_peri_recov_theta_amplitude = np.mean(peri_recov_theta_amp, axis=0)
    all_recov_theta_amplitude.append(avg_peri_recov_theta_amplitude)
        
    peri_recov_cir_dev = circvar(peri_recov_theta_phase, high=3.14159, low =-3.14159,
                                 axis=0)
    all_recov_cir_dev.append(peri_recov_cir_dev)


#%% plot theta amplitude (all)
fig, ax = plt.subplots(figsize=(3.5,3))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)
ax.set(ylabel='theta amplitude (dB)', xlabel='time (s)')
fig.suptitle('Theta amplitude, stim v non-stim')

mean_amp_stim = np.mean(all_stim_theta_amplitude, axis=0)
mean_amp_stim_db = [20 * log10(x) for x in mean_amp_stim]
mean_amp_cont = np.mean(all_cont_theta_amplitude, axis=0)
mean_amp_cont_db = [20 * log10(x) for x in mean_amp_cont]

ax.plot(taxis, mean_amp_stim_db, 'royalblue')
ax.plot(taxis, mean_amp_cont_db, 'grey', alpha=.5)

fig.tight_layout()
plt.show()

# fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_nonstim_theta_amp.png',
#             bbox_inches='tight',
#             dpi=500)

plt.close(fig)


#%% calculate and plot comparison theta amplitude (all)
amp_stims = []; amp_starts = []

for sess in all_stim_theta_amplitude:
    amp_stims.append(np.mean(sess[2500:3750]))  # 1~2 s
for sess in all_start_theta_amplitude:
    amp_starts.append(np.mean(sess[2500:3750]))
    
pval = wilcoxon(amp_stims, amp_starts)[1]


fig, ax = plt.subplots(figsize=(3,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.set_xticklabels(['non-stim', 'stim'], minor=False)
ax.set_xticks([1, 2])

ax.set(ylabel='theta amplitude')
fig.suptitle('theta amplitude, stim v non_stim, p={}'.format(round(pval, 3)))

bp = ax.bar([1, 2], [np.mean(amp_starts), np.mean(amp_stims)],
            color=['grey', 'royalblue'], edgecolor=['k','k'], width=.35)
    
ax.scatter([[1]*len(amp_starts), [2]*len(amp_stims)], [amp_starts, amp_stims], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[1]*len(amp_starts), [2]*len(amp_stims)], [amp_starts, amp_stims], zorder=2,
        color='grey', alpha=.5)

ax.set(xlim=(0.5, 2.5))

fig.tight_layout()
plt.show()

# fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_pooled_ROpeak_population_earlyvlate.pdf',
#             bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_nonstim_theta_amp_bar.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)



#%% plot circular deviation stim v non-stim
fig, axs = plt.subplots(3,1,figsize=(4,8))
fig.suptitle('avg. circ. dev., stim v non-stim')

for i in range(2):
    axs[i].set(xlabel='time (s)', ylabel='avg. circ. dev. (π)')

mean_stim_cir_dev = np.mean(all_stim_cir_dev, axis=0)
mean_start_cir_dev = np.mean(all_start_cir_dev, axis=0)
sem_stim_cir_dev = sem(all_stim_cir_dev, axis=0)
sem_start_cir_dev = sem(all_start_cir_dev, axis=0)

axs[0].plot(taxis, mean_stim_cir_dev, 'royalblue')
axs[1].plot(taxis, mean_start_cir_dev, 'grey')

axs[2].plot(taxis, mean_stim_cir_dev, 'royalblue')
axs[2].plot(taxis, mean_start_cir_dev, 'grey', alpha=.5)
axs[2].fill_between(taxis, mean_stim_cir_dev+sem_stim_cir_dev,
                           mean_stim_cir_dev-sem_stim_cir_dev,
                           color='royalblue', alpha=.3)
axs[2].fill_between(taxis, mean_start_cir_dev+sem_start_cir_dev,
                           mean_start_cir_dev-sem_start_cir_dev,
                           color='grey', alpha=.15)

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_nonstim_theta_cir_dev.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


#%%
fig, axs = plt.subplots(3,1,figsize=(4,8))
# fig.suptitle('avg. circ. dev.')
axs[0].set(xlabel='time (s)', ylabel='avg. circ. dev. (π)',
       title='stim v stim-cont')

axs[0].plot(taxis, np.mean(all_stim_cir_dev, axis=0), 'royalblue')
axs[1].plot(taxis, np.mean(all_cont_cir_dev, axis=0), 'grey')

axs[2].plot(taxis, np.mean(all_stim_cir_dev, axis=0), 'royalblue')
axs[2].plot(taxis, np.mean(all_cont_cir_dev, axis=0), 'grey', alpha=.5)

fig.tight_layout()
plt.show()
plt.close(fig)


fig, axs = plt.subplots(3,1,figsize=(4,8))
# fig.suptitle('avg. circ. dev.')
axs[0].set(xlabel='time (s)', ylabel='avg. circ. dev. (π)',
       title='stim v recov')

axs[0].plot(taxis, np.mean(all_stim_cir_dev, axis=0), 'royalblue')
axs[1].plot(taxis, np.mean(all_recov_cir_dev, axis=0), 'grey')

axs[2].plot(taxis, np.mean(all_stim_cir_dev, axis=0), 'royalblue')
axs[2].plot(taxis, np.mean(all_recov_cir_dev, axis=0), 'grey', alpha=.5)

fig.tight_layout()
plt.show()
plt.close(fig)