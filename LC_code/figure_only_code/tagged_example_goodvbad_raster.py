# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:56:55 2023

plot example cell with good and bad trials from clusters with rasters 

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
beh_par_file = sio.loadmat(pathname+pathname[-18:]+
                           '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)

tot_trial = time.shape[0]

example = 'A049r-20230120-04 clu10'

# read spikes for cell 1 (clu9)
spk = np.empty(tot_trial-1, dtype='object')
spk_bef = np.empty(tot_trial-1, dtype='object')
for trial in range(1, tot_trial):  # trial 0 is an empty trial
    spk[trial-1] = spike_time_file[time[trial,10-2]][0]
    spk_bef[trial-1] = spike_time_file[time_bef[trial,10-2]][0]
spk_all = np.empty(tot_trial-1, dtype='object')
for trial in range(tot_trial-1):
    curr_trial_trunc = np.concatenate([spk_bef[trial].reshape(-1),
                                       spk[trial].reshape(-1)])
    curr_trial_trunc = [i for i in curr_trial_trunc if i>=-1250 and i<=5000]
    spk_all[trial] = curr_trial_trunc


#%% plotting 
print('plotting rasters...')

fig, ax = plt.subplots(figsize=(5, 4))
ax.set(ylim=(-1, len(spk_all)), xlim=(-1.02, 4.02),
       title=example)
for p in ['left', 'top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])

count=0
for trial in ind_bad_beh:
    count+=1
    ax.scatter([t/1250 for t in spk_all[trial]], [count]*len(spk_all[trial]),
                s=3, color='grey')
for trial in ind_good_beh[-15:]:
    count+=1
    ax.scatter([t/1250 for t in spk_all[trial]], [count]*len(spk_all[trial]),
               s=3, color='forestgreen')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell_goodvbad.png',
            dpi=300,
            bbox_inches='tight')


spk_avg = sr_file['all tagged avg'][example][2500:8750]
spk_sem = sr_file['all tagged sem'][example][2500:8750]

fig, ax = plt.subplots(figsize=(5, 4))

ax.set(ylabel='spike rate (Hz)',
       ylim=(0, max(spk_avg)*1.2))

xaxis = np.arange(-1250, 5000)/1250

ax.plot(xaxis, spk_avg, color='grey')
ax.fill_between(xaxis, spk_avg+spk_sem, spk_avg-spk_sem,
                color='grey', alpha=.25)

# fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell.png',
#             dpi=300,
#             bbox_inches='tight')