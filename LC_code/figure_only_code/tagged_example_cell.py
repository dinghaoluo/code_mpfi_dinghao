# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:56:55 2023

plot example cells from clusters with rasters 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import h5py


#%% MAIN
# use cells from the same session! 
pathname = 'Z:\Dinghao\MiceExp\ANMD049r\A049r-20230120\A049r-20230120-04'
print('using {}'.format(pathname[-17:]))
filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']

sr_file = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_avg_sem.npy',
                  allow_pickle=True).item()

tot_trial = time.shape[0]

example1 = 'A049r-20230120-04 clu9'
example2 = 'A049r-20230120-04 clu12'

# read spikes for cell 1 (clu9)
spk1 = np.empty(tot_trial-1, dtype='object')
spk1_bef = np.empty(tot_trial-1, dtype='object')
for trial in range(1, tot_trial):  # trial 0 is an empty trial
    spk1[trial-1] = spike_time_file[time[trial,7]][0]
    spk1_bef[trial-1] = spike_time_file[time_bef[trial,7]][0]
spk1_all = np.empty(tot_trial-1, dtype='object')
for trial in range(tot_trial-1):
    curr_trial_trunc = np.concatenate([spk1_bef[trial].reshape(-1),
                                       spk1[trial].reshape(-1)])
    curr_trial_trunc = [i for i in curr_trial_trunc if i>=-1250 and i<=5000]
    spk1_all[trial] = curr_trial_trunc

# read spikes for cell 2 (clu12)    
spk2 = np.empty(tot_trial-1, dtype='object')
spk2_bef = np.empty(tot_trial-1, dtype='object')
for trial in range(1, tot_trial):  # trial 0 is an empty trial
    spk2[trial-1] = spike_time_file[time[trial,10]][0]
    spk2_bef[trial-1] = spike_time_file[time_bef[trial,10]][0]
spk2_all = np.empty(tot_trial-1, dtype='object')
for trial in range(tot_trial-1):
    curr_trial_trunc = np.concatenate([spk2_bef[trial].reshape(-1),
                                       spk2[trial].reshape(-1)])
    curr_trial_trunc = [i for i in curr_trial_trunc if i>=-1250 and i<=5000]
    spk2_all[trial] = curr_trial_trunc
    

#%% plotting 
print('plotting rasters...')

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4)); fig.tight_layout(pad=4)
ax1.set(title=example1,
        xlabel='time (s)', ylabel='trial',
        ylim=(0, 99), xlim=(-1, 4))
ax2.set(title=example2,
        xlabel='time (s)', ylabel='trial',
        ylim=(0, 99), xlim=(-1, 4))
ax1.set_yticks([1]+list(np.arange(20, 101, 20)))
ax2.set_yticks([1]+list(np.arange(20, 101, 20)))
ax1.patch.set_facecolor('lightsteelblue')
ax2.patch.set_facecolor('peachpuff')
ax1.patch.set_alpha(.3)
ax2.patch.set_alpha(.3)

for trial in range(tot_trial-1):
    ax1.scatter([t/1250 for t in spk1_all[trial]], [trial]*len(spk1_all[trial]),
                s=3, color='grey', alpha=.2)
    ax2.scatter([t/1250 for t in spk2_all[trial]], [trial]*len(spk2_all[trial]),
                s=3, color='grey', alpha=.2)

    
print('plotting avg spk profiles...')

spk1_avg = sr_file['all tagged avg'][example1][2500:8750]
spk1_sem = sr_file['all tagged sem'][example1][2500:8750]
spk2_avg = sr_file['all tagged avg'][example2][2500:8750]
spk2_sem = sr_file['all tagged sem'][example2][2500:8750]

ax1t = ax1.twinx(); ax2t = ax2.twinx()
ax1t.set(ylabel='spike rate (Hz)',
         ylim=(0, max(spk1_avg)*1.2))
ax2t.set(ylabel='spike rate (Hz)',
         ylim=(min(spk1_avg)*.5, max(spk2_avg)*1.2))

xaxis = np.arange(-1250, 5000)/1250

ax1t.plot(xaxis, spk1_avg, color='grey')
ax1t.fill_between(xaxis, spk1_avg+spk1_sem, spk1_avg-spk1_sem, 
                  color='grey', alpha=.25)
ax2t.plot(xaxis, spk2_avg, color='grey')
ax2t.fill_between(xaxis, spk2_avg+spk2_sem, spk2_avg-spk2_sem, 
                  color='grey', alpha=.25)

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell.png',
            dpi=300,
            bbox_inches='tight')