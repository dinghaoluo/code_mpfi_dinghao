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


#%% plotting 
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
import plotting_functions as pf
from common import mpl_formatting
mpl_formatting()


#%% loading
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
all_lickbr_non_stim = []; all_lickbr_stim = []  # control 
all_length_non_stim = []; all_length_stim = []  # control, 19 Sept 2024 Dinghao 
all_rew_perc_non_stim = []; all_rew_perc_stim = []  # control, 18 July 2025 

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
    ends = alignRun['trialsRun']['endLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    speeds = alignRun['trialsRun']['speed_MMsec'][0][0][0][1:]  # control 
    accels = alignRun['trialsRun']['accel_MMsecSq'][0][0][0][1:]  # control

    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:  # trials where no rewards were delivered or rewards were delivered after 16 s
            pumps[trial] = [np.nan]
    
    first_licks = []
    licks_bef_rew = []
    mean_speeds = []
    peak_speeds = []
    trial_lengths = []
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
        ps = max([sp[0] for sp in speeds[trial]])/10  # from mm/s to cm/s
        peak_speeds.append(ps)
        
        # smooth speed and get acceleration
        smoothed_speeds = [np.convolve(np.squeeze(s), gaus_speed, mode='same') for s in speeds]
        accels = [np.gradient(s) for s in smoothed_speeds]
        init_accels = [np.mean(s[:625])*10 for s in accels]  # take first .5 s for mean init accel
        
        # trial lengths 
        tl = (ends[trial]-starts[trial])/1250
        trial_lengths.append(tl)
        
    
    # stim licks 
    licks_stim = [first_licks[i-1] for i in stim if first_licks[i-1]!=0 and first_licks[i+1]!=0]
    licks_br_stim = [licks_bef_rew[i-1] for i in stim]
    
    pval = []; 
    pval_mspeeds = []; pval_pspeeds = []; pval_bef_rew = []; pval_initacc = []; pval_length = []  # controls 
    
    curr_licks_non_stim = []; curr_licks_stim = []
    # controls
    curr_mspeeds_non_stim = []; curr_mspeeds_stim = []
    curr_maccels_non_stim = []; curr_maccels_stim = []
    curr_pspeeds_non_stim = []; curr_pspeeds_stim = []
    curr_initacc_non_stim = []; curr_initacc_stim = []
    curr_lickbr_non_stim = []; curr_lickbr_stim = []
    curr_length_non_stim = []; curr_length_stim = []
    curr_rew_perc_non_stim = []; curr_rew_perc_stim = []
    
    
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
        lickbr_non_stim = [licks_bef_rew[i-1] for i in selected_non_stim]
        lickbr_stim = [licks_bef_rew[i-1] for i in stim]
        lengths_non_stim = [trial_lengths[i-1] for i in selected_non_stim]
        lengths_stim = [trial_lengths[i-1] for i in stim]
        
        curr_licks_non_stim.append(licks_non_stim)
        curr_licks_stim.append(licks_stim)
        curr_mspeeds_non_stim.append(mspeeds_non_stim)
        curr_mspeeds_stim.append(mspeeds_stim)
        curr_pspeeds_non_stim.append(pspeeds_non_stim)
        curr_pspeeds_stim.append(pspeeds_stim)
        curr_initacc_non_stim.append(initaccs_non_stim)
        curr_initacc_stim.append(initaccs_stim)
        curr_lickbr_non_stim.append(lickbr_non_stim)
        curr_lickbr_stim.append(lickbr_stim)
        curr_length_non_stim.append(lengths_non_stim)
        curr_length_stim.append(lengths_stim)
        curr_rew_perc_non_stim.append(np.sum([~np.isnan(pumps[i]) for i in selected_non_stim]) / len(selected_non_stim))
        curr_rew_perc_stim.append(np.sum([~np.isnan(pumps[i]) for i in stim]) / len(selected_non_stim))
        
        pval.append(ranksums(licks_non_stim, licks_stim)[1])
        pval_mspeeds.append(ranksums(mspeeds_non_stim, mspeeds_stim)[1])
        pval_pspeeds.append(ranksums(pspeeds_non_stim, pspeeds_stim)[1])
        pval_initacc.append(ranksums(initaccs_non_stim, initaccs_stim)[1])
        pval_bef_rew.append(ranksums(licks_br_non_stim, licks_br_stim)[1])
        pval_length.append(ranksums(lengths_non_stim, lengths_stim)[1])
        
    if stim_cond==2:
        all_licks_non_stim.append(np.median(curr_licks_non_stim))
        all_licks_stim.append(np.median(curr_licks_stim))
        all_mspeeds_non_stim.append(np.median(curr_mspeeds_non_stim))
        all_mspeeds_stim.append(np.median(curr_mspeeds_stim))
        all_pspeeds_non_stim.append(np.median(curr_pspeeds_non_stim))
        all_pspeeds_stim.append(np.median(curr_pspeeds_stim))
        all_initacc_non_stim.append(np.median(curr_initacc_non_stim))
        all_initacc_stim.append(np.median(curr_initacc_stim))
        all_lickbr_non_stim.append(np.median(curr_lickbr_non_stim))
        all_lickbr_stim.append(np.median(curr_lickbr_stim))
        all_length_non_stim.append(np.median(curr_length_non_stim))
        all_length_stim.append(np.median(lengths_stim))
        all_rew_perc_non_stim.append(np.median(curr_rew_perc_non_stim))
        all_rew_perc_stim.append(np.median(curr_rew_perc_stim))
    
    
    # licks plot 
    if comp_method == 'baseline':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}'.format(stim_cond, sessname)
    elif comp_method == 'stim_cont':
        savepath = 'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}'.format(stim_cond, sessname)
    pf.plot_box_with_scatter(licks_non_stim, licks_stim, 
                             xlabel='1st lick (s)', 
                             savepath=savepath)
    
    # mspeeds plot 
    if comp_method == 'baseline':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_velocity'.format(stim_cond, sessname)
    elif comp_method == 'stim_cont':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_velocity'.format(stim_cond, sessname)
    pf.plot_box_with_scatter(mspeeds_non_stim, mspeeds_stim, 
                             xlabel='mean speed (cm/s)', 
                             savepath=savepath,
                             show_scatter=False)
        
    # pspeeds plot 
    if comp_method == 'baseline':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_peak_velocity'.format(stim_cond, sessname)
    elif comp_method == 'stim_cont':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_peak_velocity'.format(stim_cond, sessname)
    pf.plot_box_with_scatter(pspeeds_non_stim, pspeeds_stim, 
                             xlabel='peak speed (cm/s)', 
                             savepath=savepath,
                             show_scatter=False)
        
    # init acc plot 
    if comp_method == 'baseline':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_init_accel'.format(stim_cond, sessname)
    elif comp_method == 'stim_cont':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_init_accel'.format(stim_cond, sessname)
    pf.plot_box_with_scatter(initaccs_non_stim, initaccs_stim, 
                             xlabel='init. acceleration (cm/s2)', 
                             savepath=savepath,
                             show_scatter=False)
    
    # lick-bef-rew plot 
    if comp_method == 'baseline':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_licks_bef_rew'.format(stim_cond, sessname)
    elif comp_method == 'stim_cont':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_licks_bef_rew'.format(stim_cond, sessname)
    pf.plot_box_with_scatter(lickbr_non_stim, lickbr_stim, 
                             xlabel='licks bef. rew.', 
                             savepath=savepath,
                             show_scatter=False)
    
    # trial-length plot 
    if comp_method == 'baseline':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0\{}_control_trial_length'.format(stim_cond, sessname)
    elif comp_method == 'stim_cont':
        savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_0{}0_stim_cont\{}_control_trial_length'.format(stim_cond, sessname)
    pf.plot_box_with_scatter(lengths_non_stim, lengths_stim, 
                             xlabel='trial lengths (s)', 
                             savepath=savepath,
                             show_scatter=False)
    

#%% licks summary
if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary'
pf.plot_violin_with_scatter(all_licks_non_stim, all_licks_stim, 'grey', 'royalblue',
                            paired=True,
                            title='time to 1st-licks ctrl v stim', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='t. to 1st-licks',
                            save=True, savepath=savepath, dpi=300)
    

#%% mean speeds summary
if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_velocity'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_velocity'
pf.plot_violin_with_scatter(all_mspeeds_non_stim, all_mspeeds_stim, 'grey', 'royalblue',
                            paired=True,
                            title='mean velocity ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='mean velocity',
                            save=True, savepath=savepath, dpi=300)
    

#%% peak speeds summary
if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_peak_velocity'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_peak_velocity'
pf.plot_violin_with_scatter(all_pspeeds_non_stim, all_pspeeds_stim, 'grey', 'royalblue',
                            paired=True,
                            title='peak velocity ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='peak velocity',
                            save=True, savepath=savepath, dpi=300)
    
    
#%% init accels summary
if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_init_accel'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_init_accel'
pf.plot_violin_with_scatter(all_initacc_non_stim, all_initacc_stim, 'grey', 'royalblue',
                            paired=True,
                            title='init. accel. ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='init. acceleration',
                            save=True, savepath=savepath, dpi=300)


#%% licks bef rew summary
if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_licks_bef_rew'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_licks_bef_rew'
pf.plot_violin_with_scatter(all_lickbr_non_stim, all_lickbr_stim, 'grey', 'royalblue',
                            paired=True,
                            title='licks bef. rew. ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='licks bef. rew.',
                            save=True, savepath=savepath, dpi=300)


#%% trial length summary
if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_trial_length'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_trial_length'
pf.plot_violin_with_scatter(all_length_non_stim, all_length_stim, 'grey', 'royalblue',
                            paired=True,
                            title='trial lengths ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='trial length',
                            save=True, savepath=savepath, dpi=300)


#%% reward percentage summary
# for i in range(len(all_rew_perc_non_stim)):
#     if all_rew_perc_non_stim[i] > 1 or all_rew_perc_stim[i] > 1:
#         del all_rew_perc_non_stim[i]
#         del all_rew_perc_stim[i]
#         break

if comp_method == 'baseline':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020\summary_control_rew_perc'
elif comp_method == 'stim_cont':
    savepath = r'Z:\Dinghao\code_dinghao\LC_opto_ephys\opto_licktime_020_stim_cont\summary_control_rew_perc'
pf.plot_violin_with_scatter(all_rew_perc_non_stim, all_rew_perc_stim, 'grey', 'royalblue',
                            paired=True,
                            title='rewarded trial % ctrl. v stim.', 
                            xticklabels=['ctrl.', 'stim.'],
                            ylabel='% rewarded trials',
                            ylim=(0.85,1.01),
                            save=True, savepath=savepath, dpi=300)
