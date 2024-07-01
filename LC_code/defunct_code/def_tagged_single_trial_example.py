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
import h5py
import sys

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')


#%% MAIN
# used the following session(s) for example trace plotting
pathname = 'Z:\Dinghao\MiceExp\ANMD049r\A049r-20230120\A049r-20230120-04'
filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'

print('session used: {}'.format(pathname[-17:]))

# load files 
speed_time_file = sio.loadmat(filename + '_alignRun_msess1.mat')
spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
tagged_id_file = np.load('Z:\Dinghao\code_dinghao\LC_tagged_by_sess'+
                         pathname[-18:]+'_tagged_spk.npy',
                         allow_pickle=True).item()

time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']
tag_sess = list(tagged_id_file.keys())
print('number of tagged cells in session: {}'.format(len(tag_sess)))

tot_clu = time.shape[1]
tot_trial = time.shape[0]  # trial 1 is empty but tot_trial includes it for now
    
samp_freq = 1250  # Hz
gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
sigma_speed = 12.5
 
# speed of all trials
speed_time_bef = speed_time_file['trialsRun'][0]['speed_MMsecBef'][0][0]
speed_time = speed_time_file['trialsRun'][0]['speed_MMsec'][0][0]
gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]

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

# trial length for spike-reading
trial_length = [trial.shape[0] for trial in speed_time_conv]

# spike reading
spike_time = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
spike_time_bef = np.empty(shape=(tot_clu, tot_trial - 1), dtype='object')
for i in tag_sess:
    i = int(i)-2
    for j in range(1, tot_trial):  # trial 0 is an empty trial
        spike_time[i,j-1] = spike_time_file[time[j,i]][0]
        spike_time_bef[i,j-1] = spike_time_file[time_bef[j,i]][0]
spike_train_all = np.empty(shape=(len(tag_sess), tot_trial - 1), dtype='object')
for i in range(len(tag_sess)):
    for trial in range(tot_trial-1):
        clu_id = int(tag_sess[i])-2
        spikes = np.concatenate([spike_time_bef[clu_id][trial].reshape(-1),
                                 spike_time[clu_id][trial].reshape(-1)])
        spikes = [int(s+3750) for s in spikes]
        spike_train_trial = np.zeros(trial_length[trial])
        spike_train_trial[spikes] = 1
        spike_train_all[i][trial] = spike_train_trial
        
        
#%% plotting 
trial_used = 59
print('example trial: {}'.format(trial_used))

fig, ax = plt.subplots(figsize=(6, 1.8))
ax.set(xlabel='time (s)', ylabel='velocity\n(cm/s)',
       ylim=(-5,100))
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
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egsess_good_speed.png',
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
trial_used = 45
print('example trial: {}'.format(trial_used))

fig, ax = plt.subplots(figsize=(6, 1.8))

ax.set(xlabel='time (s)', ylabel='velocity\n(cm/s)',
       ylim=(-5, 100))
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
licks_last = speed_time_file['trialsRun'][0]['lickLfpInd'][0][0][trial_used-1].reshape(-1)
licks = speed_time_file['trialsRun'][0]['lickLfpInd'][0][0][trial_used-1].reshape(-1)
# start time for this trial
startLfp = speed_time_file['trialsRun'][0]['startLfpInd'][0][0][trial_used-1]
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