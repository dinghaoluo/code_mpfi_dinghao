# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:49:59 2024

first-lick profiles 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import sys

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathOpt = rec_list.pathLCopt


#%% distance 
all_first_licks_dist = []

for pathname in pathOpt:
    sessname = pathname[-17:]
    print(sessname)
    
    infofilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    
    Info = sio.loadmat(infofilename)
    pulseMethod = Info['beh'][0][0]['pulseMethod'][0]
    
    # stim info
    use_trials = [i for i, e in enumerate(pulseMethod) if e==0]
    
    # licks 
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    dist = alignRun['trialsRun']['xMM'][0][0][0][1:]  # distance at each sample
    
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        # licks 
        if trial+1 in use_trials:
            lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
            if len(lk)!=0:  # append only if there is licks in this trial
                for i in range(len(lk)):
                    ld = dist[trial][lk[0]-starts[trial]]/10                
                    if ld > 30:  # filter out first licks before 30 (only starts counting at 30)
                        all_first_licks_dist.append(dist[trial][lk[0]-starts[trial]]/10)
                        break
                if ld <= 30:
                    all_first_licks_dist.append(0)
            else:
                all_first_licks_dist.append(0)

# squeeze array 
all_1st_licks_dist = [float(t) for t in all_first_licks_dist]

# histogram
counts, bins = np.histogram(all_1st_licks_dist, bins=120)
counts[0] = 0
counts = normalise(counts)

# plotting 
fig, ax = plt.subplots(figsize=(3,2))

ax.stairs(counts, bins, color='orchid')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(0, 220), xlabel='distance (cm)', xticks=[0, 100, 200],
       ylabel='hist. 1st-licks', yticks=[0, 1])

fig.tight_layout()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_first_lick_hist_dist.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_first_lick_hist_dist.pdf',
            dpi=500,
            bbox_inches='tight')


#%% time 
all_first_licks_time = []
all_pumps_time = []

for pathname in pathOpt:
    sessname = pathname[-17:]
    print(sessname)
    
    infofilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    
    Info = sio.loadmat(infofilename)
    pulseMethod = Info['beh'][0][0]['pulseMethod'][0]
    
    # stim info
    use_trials = [i for i, e in enumerate(pulseMethod) if e==0]
    
    # licks 
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        # licks 
        if trial+1 in use_trials:
            lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
            pp = pumps[trial]-starts[trial]
            if len(lk)!=0 and len(pp)!=0:  # append only if there is licks in this trial
                all_first_licks_time.append([lk[0]-starts[trial]])
                all_pumps_time.append(pp[0][0])

    
#%% squeeze array 
all_1st_licks_time = [t[0]/1250 for t in all_first_licks_time if t[0]<1250*8]
all_pumps_s_time = [p/1250 for p in all_pumps_time if p<1250*8]

    
#%% histogram
counts, bins = np.histogram(all_1st_licks_time, bins=120)
counts[0] = 0
counts = normalise(counts)

counts_p, bins_p = np.histogram(all_pumps_s_time, bins=120)
counts_p = normalise(counts_p)


#%% plotting 
fig, ax = plt.subplots(figsize=(3,2))

ax.stairs(counts, bins, color='orchid')
# ax.stairs(counts_p, bins_p, color='darkgreen')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xlim=(0, 5), xlabel='time (s)', xticks=[0, 2, 4],
       ylabel='hist. 1st-licks', yticks=[0, 1])

fig.tight_layout()

fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_first_lick_hist_time.png',
            dpi=500,
            bbox_inches='tight')
fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_first_lick_hist_time.pdf',
            dpi=500,
            bbox_inches='tight')