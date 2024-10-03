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
from scipy.stats import sem


#%% load data 
all_info = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                   allow_pickle=True).item()


#%% MAIN
# use cells from the same session! 
pathname = 'Z:\Dinghao\MiceExp\ANMD049r\A049r-20230120\A049r-20230120-04'
print('using {}'.format(pathname[-17:]))
filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
spike_time_file = h5py.File(filename + '_alignedSpikesPerNPerT_msess1_Run0.mat')['trialsRunSpikes']
time_bef = spike_time_file['TimeBef']; time = spike_time_file['Time']

beh_par_file = sio.loadmat(pathname+pathname[-18:]+
                           '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)

tot_trial = time.shape[0]

example = 'A049r-20230120-04 clu8'
example_trains = all_info[example]

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
    # curr_trial_trunc = [i for i in curr_trial_trunc if i>=-725 and i<=1250]
    spk_all[trial] = curr_trial_trunc


#%% plotting 
print('plotting rasters...')

fig, ax = plt.subplots(figsize=(4, 3))
ax.set(ylim=(-1, len(spk_all)), xlim=(-1, 4),
       title=example)
for p in ['left', 'top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])

count=0
for trial in ind_bad_beh:
    count+=1
    ax.scatter([t/1250 for t in spk_all[trial]], [count]*len(spk_all[trial]),
                s=3, color='darkgrey')
for trial in ind_good_beh[-15:]:
    count+=1
    ax.scatter([t/1250 for t in spk_all[trial]], [count]*len(spk_all[trial]),
               s=3, color='limegreen')

plt.show()
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell_goodvbad_-1-4.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell_goodvbad_-1-4.pdf',
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\paper\figures\figure_1_LC_tagged_egcell_goodvbad_-1-4.pdf',
            bbox_inches='tight')

plt.close(fig)


good_trains = [t[2500:8750]*1250 for i,t in enumerate(example_trains) if i in ind_good_beh if len(t)>=8750]
bad_trains = [t[2500:8750]*1250 for i,t in enumerate(example_trains) if i in ind_bad_beh if len(t)>=8750]

good_avg = np.mean(good_trains, axis=0)
good_sem = sem(good_trains, axis=0)
bad_avg = np.mean(bad_trains, axis=0)
bad_sem = sem(bad_trains, axis=0)

fig, ax = plt.subplots(figsize=(2,1.6))

ax.set(xlabel='time (s)',
       ylabel='spike rate (Hz)',
       xlim=(-1, 4),
       ylim=(0, max(good_avg)*1.2),
       xticks=[0,2,4],
       yticks=[0,4])

xaxis = np.arange(-1250, 5000)/1250

gd, = ax.plot(xaxis, good_avg, color='limegreen')
ax.fill_between(xaxis, good_avg+good_sem, good_avg-good_sem,
                color='limegreen', alpha=.25)

bd, = ax.plot(xaxis, bad_avg, color='darkgrey')
ax.fill_between(xaxis, bad_avg+bad_sem, bad_avg-bad_sem,
                color='darkgrey', alpha=.25)

ax.legend([gd, bd], ['good trial', 'bad trial'], frameon=False, fontsize=8)

for p in ['top', 'right']:
    ax.spines[p].set_visible(False)

plt.show()
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell_-1-4.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_egcell_-1_4.pdf',
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\paper\figures\figure_1_LC_tagged_egcell_-1_4.pdf',
            bbox_inches='tight')

plt.close(fig)