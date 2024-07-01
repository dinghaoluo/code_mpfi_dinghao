# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 13:26:35 2023

compare opto stim vs baseline licktime

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats import ranksums, wilcoxon, ttest_rel 
import sys

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathOpt = rec_list.pathLCopt

# 0-2-0
sess_list = [sess[-17:] for sess in pathOpt]

n_bst = 1000  # hyperparameter for bootstrapping
comp_method = 'baseline'
print('\n**BOOTSTRAP # = {}**'.format(n_bst))
print('**comparison method: {}**\n'.format(comp_method))

samp_freq = 1250  # Hz
gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
sigma_speed = samp_freq/100  # 10 ms
gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]


#%% MAIN 
all_licks_non_stim = []; all_licks_stim = []
all_mspeeds_non_stim = []; all_mspeeds_stim = []  # control 
all_accel_non_stim = []; all_accel_stim = []  # control 
all_pspeeds_non_stim = []; all_pspeeds_stim = []  # control 
all_initacc_non_stim = []; all_initacc_stim = []  # control 

for sessname in sess_list:
    print(sessname)
    
    infofilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    
    Info = sio.loadmat(infofilename)
    pulseMethod = Info['beh'][0][0]['pulseMethod'][0]
    
    # stim info
    tot_stims = len([t for t in pulseMethod if t!=0])
    stim_cond = pulseMethod[np.where(pulseMethod!=0)][0]  # check stim condition
    stim = [i for i, e in enumerate(pulseMethod) if e==stim_cond]
    
    # licks 
    lickfilename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname[1:5], sessname[:14], sessname[:17], sessname[:17])
    alignRun = sio.loadmat(lickfilename)
    
    # ignore all 1st trials since it is before counting starts and is an empty cell
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    speeds = alignRun['trialsRun']['speed_MMsec'][0][0][0][1:]  # control 
    accels = alignRun['trialsRun']['accel_MMsecSq'][0][0][0][1:]  # control

    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:  # trials where no rewards were delivered or rewards were delivered after 16 s
            pumps[trial] = 20000
    
    first_licks = []
    licks_bef_rew = []
    mean_speeds = []
    peak_speeds = []
    for trial in range(tot_trial):

        # licks 
        lk = [l[0] for l in licks[trial] if l-starts[trial] > 1250]  # exclude licks in the 1st second, as they could be carry-over licks from the last trial
        if len(lk)!=0:  # append only if there is licks in this trial
            first_licks.append((lk[0]-starts[trial])/1250)
        else:
            first_licks.append(0)
        licks_bef_rew.append(len([l for l in lk if l-starts[trial]<pumps[trial]]))

        # mean speed
        ms = np.mean(speeds[trial])/10  # from mm/s to cm/s
        mean_speeds.append(ms)
        
        # peak speed
        ps = max(speeds[trial])/10  # from mm/s to cm/s
        peak_speeds.append(ps)
        
        # smooth speed and get acceleration
        smoothed_speeds = [np.convolve(np.squeeze(s), gaus_speed, mode='same') for s in speeds]
        accels = [np.gradient(s) for s in smoothed_speeds]
        init_accels = [np.mean(s[:625])*10 for s in accels]  # take first .5 s for mean init accel
        
    
    # stim licks 
    licks_stim = [first_licks[i-1] for i in stim if first_licks[i-1]!=0 and first_licks[i+1]!=0]
    licks_br_stim = [licks_bef_rew[i-1] for i in stim]
    
    pval = []; 
    pval_mspeeds = []; pval_pspeeds = []; pval_bef_rew = []; pval_initacc = []  # controls 
    
    curr_licks_non_stim = []; curr_licks_stim = []
    # controls
    curr_mspeeds_non_stim = []; curr_mspeeds_stim = []
    curr_maccels_non_stim = []; curr_maccels_stim = []
    curr_pspeeds_non_stim = []; curr_pspeeds_stim = []
    curr_initacc_non_stim = []; curr_initacc_stim = []
    
    
    for i in range(n_bst):
        # select same number of non_stim to match 
        non_stim_trials = np.where(pulseMethod==0)[0]
        if comp_method == 'baseline':
            selected_non_stim = non_stim_trials[np.random.randint(0, stim[0]-1, len(licks_stim))]
            licks_non_stim = [first_licks[i-1] for i in selected_non_stim]
            licks_br_non_stim = [licks_bef_rew[i-1] for i in selected_non_stim]
        elif comp_method == 'stim_cont': # stim_control 
            selected_non_stim = [i+2 for i in stim]
            licks_non_stim = [first_licks[i-1] for i in selected_non_stim if first_licks[i-1]!=0 and first_licks[i-3]!=0]
            licks_br_non_stim = [licks_bef_rew[i-1] for i in selected_non_stim]

        mspeeds_non_stim = [mean_speeds[i-1] for i in selected_non_stim]
        mspeeds_stim = [mean_speeds[i-1] for i in stim]
        
        pspeeds_non_stim = [peak_speeds[i-1] for i in selected_non_stim]
        pspeeds_stim = [peak_speeds[i-1] for i in stim]
        
        initaccs_non_stim = [init_accels[i-1] for i in selected_non_stim]
        initaccs_stim = [init_accels[i-1] for i in stim]
        
        curr_licks_non_stim.append(licks_non_stim)
        curr_licks_stim.append(licks_stim)

        curr_mspeeds_non_stim.append(mspeeds_non_stim)
        curr_mspeeds_stim.append(mspeeds_stim)
        
        curr_pspeeds_non_stim.append(pspeeds_non_stim)
        curr_pspeeds_stim.append(pspeeds_stim)
        
        curr_initacc_non_stim.append(initaccs_non_stim)
        curr_initacc_stim.append(initaccs_stim)
        
        pval.append(ranksums(licks_non_stim, licks_stim)[1])
        pval_mspeeds.append(ranksums(mspeeds_non_stim, mspeeds_stim)[1])
        pval_pspeeds.append(ranksums(pspeeds_non_stim, pspeeds_stim)[1])
        pval_initacc.append(ranksums(initaccs_non_stim, initaccs_stim)[1])
        pval_bef_rew.append(ranksums(licks_br_non_stim, licks_br_stim)[1])
        
    if stim_cond==2:
        all_licks_non_stim.append(np.median(curr_licks_non_stim))
        all_licks_stim.append(np.median(curr_licks_stim))
        all_mspeeds_non_stim.append(np.median(curr_mspeeds_non_stim))
        all_mspeeds_stim.append(np.median(curr_mspeeds_stim))
        all_pspeeds_non_stim.append(np.median(curr_pspeeds_non_stim))
        all_pspeeds_stim.append(np.median(curr_pspeeds_stim))
        all_initacc_non_stim.append(np.median(curr_initacc_non_stim))
        all_initacc_stim.append(np.median(curr_initacc_stim))
    
    
    # licks plot 
    fig, ax = plt.subplots(figsize=(4,3))
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}, p={}, {}'.format(sessname, stim_cond, round(np.mean(pval), 4), round(np.mean(pval_bef_rew), 4)),
           ylim=(0, 1.5),
           xlabel='t 1st lick (s)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(licks_non_stim, [.5]*len(licks_stim), color='grey')
    ax.scatter(licks_stim, [1]*len(licks_stim), color='darkblue')
    ax.plot([np.median(licks_non_stim), np.median(licks_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('t 1st licks')
    if comp_method == 'baseline':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
    elif comp_method == 'stim_cont':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
    
    # mspeeds plot 
    fig, ax = plt.subplots(figsize=(4,3))
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}, p={}'.format(sessname, stim_cond, round(np.mean(pval_mspeeds), 4)),
           ylim=(0, 1.5),
           xlabel='mean velocity (cm/s)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(mspeeds_non_stim, [.5]*len(mspeeds_non_stim), color='grey')
    ax.scatter(mspeeds_stim, [1]*len(mspeeds_stim), color='darkblue')
    ax.plot([np.median(mspeeds_non_stim), np.median(mspeeds_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('mean velocity')
    
    if comp_method == 'baseline':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_velocity.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
    elif comp_method == 'stim_cont':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_velocity.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
        
    # pspeeds plot 
    fig, ax = plt.subplots(figsize=(4,3))
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}, p={}'.format(sessname, stim_cond, round(np.mean(pval_pspeeds), 4)),
           ylim=(0, 1.5),
           xlabel='peak velocity (cm/s)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(pspeeds_non_stim, [.5]*len(pspeeds_non_stim), color='grey')
    ax.scatter(pspeeds_stim, [1]*len(pspeeds_stim), color='darkblue')
    ax.plot([np.median(pspeeds_non_stim), np.median(pspeeds_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('peak velocity')
    
    if comp_method == 'baseline':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_peak_velocity.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
    elif comp_method == 'stim_cont':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_peak_velocity.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
        
    # init acc plot 
    fig, ax = plt.subplots(figsize=(4,3))
    for p in ['top', 'right', 'left']:
        ax.spines[p].set_visible(False)
    ax.set(title='{}, stim={}, p={}'.format(sessname, stim_cond, round(np.mean(pval_initacc), 4)),
           ylim=(0, 1.5),
           xlabel=u'init. accel. (cm/s\u00b2)')
    ax.set_yticks([.5, 1])
    ax.set_yticklabels(['baseline', 'stim'])
    ax.scatter(initaccs_non_stim, [.5]*len(initaccs_non_stim), color='grey')
    ax.scatter(initaccs_stim, [1]*len(initaccs_stim), color='darkblue')
    ax.plot([np.median(initaccs_non_stim), np.median(initaccs_stim)], [.5, 1],
            color='grey', alpha=.5)
    fig.suptitle('initial acceleration')
    
    if comp_method == 'baseline':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_init_accel.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')
    elif comp_method == 'stim_cont':
        fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_init_accel.png'.format(stim_cond, sessname),
                    dpi=300,
                    bbox_inches='tight')


#%% licks summary
wilc_p = wilcoxon(all_licks_non_stim, all_licks_stim)[1]
ttest_p = ttest_rel(all_licks_non_stim, all_licks_stim)[1]

# licks summary
fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([all_licks_non_stim, all_licks_stim],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

ax.scatter([1.1]*len(all_licks_non_stim), 
           all_licks_non_stim, 
           s=10, c='grey', ec='none', lw=.5, alpha=.2)
ax.scatter([1.9]*len(all_licks_stim), 
           all_licks_stim, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.2)
ax.plot([[1.1]*len(all_licks_stim), [1.9]*len(all_licks_stim)], [all_licks_non_stim, all_licks_stim], 
        color='grey', alpha=.2, linewidth=1)

ax.plot([1.1, 1.9], [np.median(all_licks_non_stim), np.median(all_licks_stim)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.median(all_licks_non_stim), 
           s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9, np.median(all_licks_stim), 
           s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ymin = min(min(all_licks_stim), min(all_licks_non_stim))-.2
ymax = max(max(all_licks_stim), max(all_licks_non_stim))+.2
ax.set(xlim=(.5,2.5), ylim=(ymin,ymax),
       yticks=[3,5],
       ylabel='time 1st licks (s)',
       title='time 1st licks ctrl v stim\nwilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl.', 'stim.'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

if comp_method == 'baseline':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary.pdf',
                bbox_inches='tight')
elif comp_method == 'stim_cont':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary.pdf',
                bbox_inches='tight')
    

#%% mean speeds summary
wilc_p_mspeeds = wilcoxon(all_mspeeds_non_stim, all_mspeeds_stim)[1]
ttest_p_mspeeds = ttest_rel(all_mspeeds_non_stim, all_mspeeds_stim)[1]

fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([all_mspeeds_non_stim, all_mspeeds_stim],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

ax.scatter([1.1]*len(all_mspeeds_non_stim), 
           all_mspeeds_non_stim, 
           s=10, c='grey', ec='none', lw=.5, alpha=.2)
ax.scatter([1.9]*len(all_mspeeds_stim), 
           all_mspeeds_stim, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.2)
ax.plot([[1.1]*len(all_mspeeds_non_stim), [1.9]*len(all_mspeeds_stim)], [all_mspeeds_non_stim, all_mspeeds_stim], 
        color='grey', alpha=.2, linewidth=1)

ax.plot([1.1, 1.9], [np.median(all_mspeeds_non_stim), np.median(all_mspeeds_stim)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.median(all_mspeeds_non_stim), 
           s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9, np.median(all_mspeeds_stim), 
           s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ymin = min(min(all_mspeeds_stim), min(all_mspeeds_non_stim))-2
ymax = max(max(all_mspeeds_stim), max(all_mspeeds_non_stim))+2
ax.set(xlim=(.5,2.5), ylim=(ymin,ymax), yticks=[30,40,50],
       ylabel='mean velocity (cm/s)',
       title='mean vel. ctrl v stim\nwilc_p={}\nttest_p={}'.format(round(wilc_p_mspeeds, 5), round(ttest_p_mspeeds, 5)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl.', 'stim.'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

plt.show()
if comp_method == 'baseline':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_velocity.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_velocity.pdf',
                bbox_inches='tight')
elif comp_method == 'stim_cont':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_velocity.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_velocity.pdf',
                bbox_inches='tight')
    
plt.close(fig)
    

#%% peak speeds summary
wilc_p_pspeeds = wilcoxon(all_pspeeds_non_stim, all_pspeeds_stim)[1]
ttest_p_pspeeds = ttest_rel(all_pspeeds_non_stim, all_pspeeds_stim)[1]

fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([all_pspeeds_non_stim, all_pspeeds_stim],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

ax.scatter([1.1]*len(all_pspeeds_non_stim), 
           all_pspeeds_non_stim, 
           s=10, c='grey', ec='none', lw=.5, alpha=.2)
ax.scatter([1.9]*len(all_pspeeds_stim), 
           all_pspeeds_stim, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.2)
ax.plot([[1.1]*len(all_pspeeds_non_stim), [1.9]*len(all_pspeeds_stim)], [all_pspeeds_non_stim, all_pspeeds_stim], 
        color='grey', alpha=.2, linewidth=1)

ax.plot([1.1, 1.9], [np.median(all_pspeeds_non_stim), np.median(all_pspeeds_stim)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.median(all_pspeeds_non_stim), 
           s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9, np.median(all_pspeeds_stim), 
           s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ymin = min(min(all_pspeeds_stim), min(all_pspeeds_non_stim))-2
ymax = max(max(all_pspeeds_stim), max(all_pspeeds_non_stim))+2
ax.set(xlim=(.5,2.5), ylim=(ymin,ymax), yticks=[60,80,100],
       ylabel='peak velocity (cm/s)',
       title='peak vel. ctrl v stim\nwilc_p={}\nttest_p={}'.format(round(wilc_p_pspeeds, 5), round(ttest_p_pspeeds, 5)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl.', 'stim.'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

plt.show()
if comp_method == 'baseline':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_peak_velocity.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_peak_velocity.pdf',
                bbox_inches='tight')
elif comp_method == 'stim_cont':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_peak_velocity.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_peak_velocity.pdf',
                bbox_inches='tight')
    
plt.close(fig)
    
    
#%% init accels summary
wilc_p_accel = wilcoxon(all_initacc_non_stim, all_initacc_stim)[1]
ttest_p_accel = ttest_rel(all_initacc_non_stim, all_initacc_stim)[1]

fig, ax = plt.subplots(figsize=(2,3))

vp = ax.violinplot([all_initacc_non_stim, all_initacc_stim],
                   positions=[1, 2],
                   showextrema=False)

vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('royalblue')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
    vp['bodies'][i].set_alpha(.75)
    b = vp['bodies'][i]
    # get the centre 
    m = np.mean(b.get_paths()[0].vertices[:,0])
    # make paths not go further right/left than the centre 
    if i==0:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
    if i==1:
        b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

ax.scatter([1.1]*len(all_initacc_non_stim), 
           all_initacc_non_stim, 
           s=10, c='grey', ec='none', lw=.5, alpha=.2)
ax.scatter([1.9]*len(all_initacc_stim), 
           all_initacc_stim, 
           s=10, c='royalblue', ec='none', lw=.5, alpha=.2)
ax.plot([[1.1]*len(all_initacc_non_stim), [1.9]*len(all_initacc_stim)], [all_initacc_non_stim, all_initacc_stim], 
        color='grey', alpha=.2, linewidth=1)

ax.plot([1.1, 1.9], [np.median(all_initacc_non_stim), np.median(all_initacc_stim)],
        color='grey', linewidth=2)
ax.scatter(1.1, np.median(all_initacc_non_stim), 
           s=30, c='grey', ec='none', lw=.5, zorder=2)
ax.scatter(1.9, np.median(all_initacc_stim), 
           s=30, c='royalblue', ec='none', lw=.5, zorder=2)
ymin = min(min(all_initacc_stim), min(all_initacc_non_stim))-.5
ymax = max(max(all_initacc_stim), max(all_initacc_non_stim))+.5
ax.set(xlim=(.5,2.5), ylim=(ymin,ymax), yticks=[2,5,8],
       ylabel='init. accel. (cm/s\u00b2)',
       title='init. accel. ctrl v stim\nwilc_p={}\nttest_p={}'.format(round(wilc_p_accel, 5), round(ttest_p_accel, 5)))
ax.set_xticks([1, 2]); ax.set_xticklabels(['ctrl.', 'stim.'])
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)

plt.show()
if comp_method == 'baseline':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_init_accel.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_init_accel.pdf',
                bbox_inches='tight')
elif comp_method == 'stim_cont':
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_init_accel.png',
                dpi=500,
                bbox_inches='tight')
    fig.savefig('Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_init_accel.pdf',
                bbox_inches='tight')
    
plt.close(fig)