# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:22:11 2023

For presentations: single-trial speed and multiple tagged-clusters plotting
Example session used: ANMD049r-20230120-04

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% MAIN
# used the following session(s) for example trace plotting
pathname = 'Z:\Dinghao\MiceExp\ANMD049r\A049r-20230120\A049r-20230120-04'
filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'

print('session used: {}'.format(pathname[-17:]))

# load files 
alignCue = sio.loadmat(filename+'_alignCue_msess1.mat')
alignRun = sio.loadmat(filename+'_alignRun_msess1.mat')

# import bad beh trial numbers
beh_par_file = sio.loadmat(pathname+pathname[-18:]+
                           '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference    
samp_freq = 1250  # Hz
gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
sigma_speed = 12.5
 
# speed of all trials
speed_time_bef = alignRun['trialsRun'][0]['speed_MMsecBef'][0][0][1:]
speed_time = alignRun['trialsRun'][0]['speed_MMsec'][0][0][1:]
gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]

cues = alignCue['trialsCue']['startLfpInd'][0][0][0][1:]
starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
diffs = cues - starts

# concatenate bef and after running onset, and convolve with gaus_speed
speed_time_all = np.empty(shape=speed_time.shape[0], dtype='object')
for i in range(speed_time.shape[0]):
    bef = speed_time_bef[i]; bef = np.zeros(3750); aft = np.squeeze(speed_time[i])
    speed_time_all[i] = np.concatenate([bef, aft])
    for tbin in range(len(speed_time_all[i])):
        if speed_time_all[i][tbin]<0:
            speed_time_all[i][tbin] = (speed_time_all[i][tbin-1])+(speed_time_all[i][tbin+1])/2
speed_time_conv = [np.convolve(np.squeeze(single), gaus_speed)[50:-49]/10 
                   for single in speed_time_all]
        
        
#%% plotting 
trial_used = 64

offset = diffs[trial_used]

print('example trial: {}'.format(trial_used))

fig, ax = plt.subplots(figsize=(5, 2))
ax.set(xlabel='time (s)', ylabel='velocity (cm/s)',
       ylim=(-3,120), xlim=(-.5, 3.5),
       xticks=[0, 1, 2, 3],
       yticks=[0, 50, 100])
for p in ['right','top']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)

xaxis = np.arange(-3750-offset, 5000-offset)/1250

ax.plot(xaxis, 
        speed_time_conv[trial_used][:8750],
        color='grey')
ax.plot([-offset/1250, -offset/1250], [-5, 100], c='r', alpha=.75, lw=3)
ax.plot([0, 0], [-5, 100], alpha=1, lw=3)

fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\start_cue_egtrial.png',
            dpi=500,
            bbox_inches='tight')
plt.close(fig)


#%%
fig, ax = plt.subplots(figsize=(6, 1.5))

ax.set(xlabel='time (s)',
       xlim=(-1, 4), ylim=(0, 2))
ax.tick_params(left=False, labelleft=False)
for p in ['right','top','left']:
    ax.spines[p].set_visible(False)
for p in ['bottom']:
    ax.spines[p].set_linewidth(1)

# read lick times for this trial (+1 since the first trial is empty)
licks_last = speed_time_file['trialsRun'][0]['lickLfpInd'][0][0][trial_used].reshape(-1)
licks = speed_time_file['trialsRun'][0]['lickLfpInd'][0][0][trial_used+1].reshape(-1)
# start time for this trial
startLfp = speed_time_file['trialsRun'][0]['startLfpInd'][0][0][trial_used+1]
licks_last = [(l-startLfp)/1250 for l in licks_last]
licks = [(l-startLfp)/1250 for l in licks]

ax.vlines(licks_last, 
          0.9, 1.1,
          color='orchid')
ax.vlines(licks, 
          0.9, 1.1,
          color='orchid')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egsess_good_licks.png',
            dpi=500,
            bbox_inches='tight')


#%%
trial_used = 60
print('example trial: {}'.format(trial_used))

fig, ax = plt.subplots(figsize=(6, 1.8))

ax.set(xlabel='time (s)', ylabel='velocity\n(cm/s)',
       ylim=(-1, 100))
for p in ['right','top']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)

xaxis = np.arange(-1250, 5000)/1250

ax.plot(xaxis, 
        speed_time_conv[trial_used][2500:8750],
        color='grey')

fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egsess_bad_speed.png',
            dpi=500,
            bbox_inches='tight')
plt.close(fig)


#%%
fig, ax = plt.subplots(figsize=(6, 1.5))

ax.set(xlabel='time (s)',
       xlim=(-1, 4), ylim=(0, 2))
ax.tick_params(left=False, labelleft=False)
for p in ['right','top','left']:
    ax.spines[p].set_visible(False)
for p in ['bottom']:
    ax.spines[p].set_linewidth(1)

# read lick times for this trial (+1 since the first trial is empty)
licks_last = speed_time_file['trialsRun'][0]['lickLfpInd'][0][0][trial_used].reshape(-1)
licks = speed_time_file['trialsRun'][0]['lickLfpInd'][0][0][trial_used+1].reshape(-1)
# start time for this trial
startLfp = speed_time_file['trialsRun'][0]['startLfpInd'][0][0][trial_used+1]
licks_last = [(l-startLfp)/1250 for l in licks_last]
licks = [(l-startLfp)/1250 for l in licks]

ax.vlines(licks_last, 
          0.9, 1.1,
          color='orchid')
ax.vlines(licks, 
          0.9, 1.1,
          color='orchid')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egsess_bad_licks.png',
            dpi=500,
            bbox_inches='tight')


#%%
fig, ax = plt.subplots(figsize=(6, 2))

ax.set(xlabel='time (s)', ylabel='cell number',
       ylim=(0.8, 2.2))
ax.spines['left'].set_visible(False)
ax.tick_params(left=False, labelleft=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# manually ordered to grant better visibility of 2 distinct groups of cells 
plot_order = [2, 3, 4, 6, 7, 0, 1, 5, 8]

baseline = 0
for i in plot_order:
    raster_curr = spike_train_all[i][trial_used][2500:8750]
    for timebin in range(len(raster_curr)):
        if raster_curr[timebin] == 0:
            raster_curr[timebin] = -1
    ax.scatter(xaxis, 
                raster_curr+baseline,
                s=1,
                color='k')
    baseline += 0.1
    
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egsess_bad_spikes.png',
            dpi=300,
            bbox_inches='tight')