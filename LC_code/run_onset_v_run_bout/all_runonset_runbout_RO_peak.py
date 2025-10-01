# -*- coding: utf-8 -*-
"""
Created on Sun June 11 13:12:54 2023
Modified on Wed 26 Feb 2025 to improve stability, fix bugs and install wrapper 
Modified on Mon 21 Apr 2025 to improve readability and include speed-matching 
    examples

LC: visual and statistical comparison between run-onset and run-bout-onset LC
    burst amplitudes

@author: Dinghao Luo
"""

#%% imports
import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import sem, wilcoxon

import plotting_functions as pf
from common import mpl_formatting, smooth_convolve
mpl_formatting()

import rec_list
recs = [path[-17:] for path in rec_list.pathLC if path[-17:] != 'A065r-20230728-02']


#%% parameters 
SAMP_FREQ = 1250  # Hz 
SAMP_FREQ_RUNBOUT = 400  # Hz 
RUN_ONSET_BIN = 3750
RUN_ONSET_BIN_RUNBOUT = 1200
BEF = 1  # x s before run/run-bout onsets 
AFT = 4  # same as above 

XAXIS_SPEED = np.arange(-1*SAMP_FREQ, 1*SAMP_FREQ) / SAMP_FREQ

run_onset_xaxis = np.arange(
    -BEF*SAMP_FREQ, AFT*SAMP_FREQ
    ) / SAMP_FREQ
run_bout_xaxis = np.arange(
    -BEF*SAMP_FREQ_RUNBOUT, AFT*SAMP_FREQ_RUNBOUT
    ) / SAMP_FREQ_RUNBOUT


#%% load data 
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )
behaviour = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\behaviour\all_LC_sessions.pkl'
    )


#%% specify RO peaking Dbh cells
tagged_RO_peak_keys = cell_profiles.index[
    (cell_profiles['run_onset_peak']) & (cell_profiles['identity']=='tagged')
    ].tolist()
putative_RO_peak_keys = cell_profiles.index[
    (cell_profiles['run_onset_peak']) & (cell_profiles['identity']=='putative')
    ].tolist()


#%% MAIN
def main(keys, list_identity):
    print(f'processing {list_identity}...')
    
    # for example speed curve plotting, 21 Apr 2025
    run_onset_speeds = []  # -1~4 s around onsets, for plotting 
    run_bout_speeds = []  # same as above 
    run_bout_speeds_matched = []  # same as above, but speed-matched 
    
    # mean curve
    mean_run_onset = []
    mean_run_bout = []
    
    # signal to noise defined as -.25 ~ .25 divided by -1~-.5 & .5~1
    s2n_run_onset = []
    s2n_run_bout = []
    
    # peak rate 
    peak_run_onset = []
    peak_run_bout = []
    
    RO_run_onset = []
    RO_run_bout = []
    
    # holder, so that we don't have to reprocess beh when changing cell
    recname = ''
    
    for cluname in keys:
        
        temp_recname = cluname[:cluname.find(' ')]
        
        clunum = int(cluname.split('clu')[1])-2  # index for retrieving fsa
        
        if temp_recname not in recs:
            continue
        
        # if new recording session, load new trains file + process beh
        if temp_recname != recname:
            recname = cluname[:cluname.find(' ')]
            
            print(f'\nprocessing {recname}...')
            
            all_trains = np.load(
                rf'Z:/Dinghao/code_dinghao/LC_ephys/all_sessions/{recname}/{recname}_all_trains_run.npy',
                allow_pickle=True).item()
            
            pathname = (
                rf'Z:\Dinghao\MiceExp\ANMD{cluname[1:5]}'
                rf'\{cluname[:14]}\{cluname[:17]}'
                )
            
            # get trial types 
            beh = behaviour.loc[recname]
            stim_conds = [trial[15] for trial in beh['trial_statements']][1:]  # index 15 is the stim condition
            stim_idx = [trial for trial, cond in enumerate(stim_conds)
                        if cond!='0']
            
            # load bad trial indices
            behPar = sio.loadmat(
                rf'{pathname}{pathname[42:]}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
                )  # -1 to account for MATLAB Python difference
            bad_idx = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                 # -1 to account for 0 being an empty trial
            good_idx = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
            good_idx = [t for t in good_idx 
                        if t not in stim_idx and t not in bad_idx]
            
            speed_f = sio.loadmat(
                rf'{pathname}{pathname[-18:]}_DataStructure_mazeSection1'
                '_TrialType1_alignRun_msess1.mat'
                )
            
            #### speed matching ####
            run_onset_speed_bef = []
            run_onset_speed_aft = []
            sess_run_onset_speeds = []
            sess_run_bout_speeds = []
            sess_run_bout_speeds_matched = []
            for idx in good_idx:
                # get the before run-onset speed mean (-1~0 s)
                curr_bef_speed = (
                    speed_f['trialsRun'][0][0]['speed_MMsecBef'][0][idx + 1]
                    [RUN_ONSET_BIN - BEF * SAMP_FREQ :]
                    )
                curr_bef_speed = [
                    round(s[0]/10) for s in curr_bef_speed
                    ]  # convert from mm/s to cm/s
                for tbin in range(1250):
                    # fix any negative value that may pop up by interpolation
                    if curr_bef_speed[tbin] < 0:
                        if tbin == 1249:
                            curr_bef_speed[tbin] = curr_bef_speed[tbin-1]
                        elif tbin == 0:
                            curr_bef_speed[tbin] = curr_bef_speed[tbin+1]
                        else:
                            curr_bef_speed[tbin] = (curr_bef_speed[tbin-1]+curr_bef_speed[tbin+1])/2
                run_onset_speed_bef.append(np.mean(curr_bef_speed))
                
                # get the aft run-onset speed mean (0~1 s)
                # get 0~4 s first, and below take mean over the first second
                curr_aft_speed = (
                    speed_f['trialsRun'][0][0]['speed_MMsec'][0][idx + 1]
                    [: AFT * SAMP_FREQ]
                    )
                curr_aft_speed = [
                    round(s[0]/10) for s in curr_aft_speed
                    ]  # same as above--conversion 
                run_onset_speed_aft.append(np.mean(curr_aft_speed[:SAMP_FREQ]))
                
                # get the curves 
                if len(curr_aft_speed) < AFT * SAMP_FREQ:
                    temp = np.zeros(AFT * SAMP_FREQ)
                    temp[:len(curr_aft_speed)] = curr_aft_speed
                    curr_aft_speed = temp
                curr_run_onset_speed=np.concatenate(
                    (curr_bef_speed, curr_aft_speed)
                    ).flatten()
                run_onset_speeds.append(curr_run_onset_speed)
                sess_run_onset_speeds.append(curr_run_onset_speed)
        
            run_onset_speed_bef_mean = np.mean(run_onset_speed_bef)
            run_onset_speed_bef_std = np.std(run_onset_speed_bef)
            run_onset_speed_aft_mean = np.mean(run_onset_speed_aft)
            run_onset_speed_aft_std = np.std(run_onset_speed_aft)
            
            # get ranges for bef and after for speed-matching
            run_onset_speed_bef_range = [
                run_onset_speed_bef_mean - 2 * run_onset_speed_bef_std,
                run_onset_speed_bef_mean + 2 * run_onset_speed_bef_std
                ]
            run_onset_speed_aft_range = [
                run_onset_speed_aft_mean - 2 * run_onset_speed_aft_std,
                run_onset_speed_aft_mean + 2 * run_onset_speed_aft_std
                ]
            
            # import beh file
            run_bout_file = mat73.loadmat(
                r'Z:\Dinghao\code_dinghao\run_bouts\fsa_run_bouts'
                rf'\{pathname[-18:]}_BefRunBout0.mat'
                )
            run_bout_table = pd.read_csv(
                r'Z:\Dinghao\code_dinghao\run_bouts'
                rf'\{pathname[-18:]}_run_bouts_py.csv'
                )
            run_bout_starts = list(run_bout_table.iloc[:,1])  # start lfp idx
            speed_all = mat73.loadmat(
                rf'{pathname}{pathname[-18:]}_BehavElectrDataLFP.mat'
                )['Track']['speed_MMsec']
        
            matched_bouts = []
            for i in range(len(run_bout_starts)):
                start = run_bout_starts[i]
                curr_bef_runbout_speed = speed_all[start - BEF * SAMP_FREQ : start]
                curr_bef_runbout_speed = [round(s/10) for s in curr_bef_runbout_speed]
                curr_aft_runbout_speed = speed_all[start : start + AFT * SAMP_FREQ]  # get 0~4 s first, same as above 
                curr_aft_runbout_speed = [round(s/10) for s in curr_aft_runbout_speed]
                
                for tbin in range(len(curr_aft_runbout_speed)):
                    # fix any negative value that may pop up by interpolation
                    if curr_aft_runbout_speed[tbin] < 0:
                        if tbin == len(curr_aft_runbout_speed)-1:
                            curr_aft_runbout_speed[tbin] = curr_aft_runbout_speed[tbin-1]
                        elif tbin == 0:
                            curr_aft_runbout_speed[tbin] = curr_aft_runbout_speed[tbin+1]
                        else:
                            curr_aft_runbout_speed[tbin] = (
                                curr_aft_runbout_speed[tbin-1] + 
                                curr_aft_runbout_speed[tbin+1]
                                ) / 2
                
                curr_speed = np.concatenate(
                    (curr_bef_runbout_speed, curr_aft_runbout_speed)
                    ).flatten()
                run_bout_speeds.append(curr_speed)
                sess_run_bout_speeds.append(curr_speed)
                
                temp_bef = np.mean(curr_bef_runbout_speed)
                temp_aft = np.mean(curr_aft_runbout_speed[: 1 * SAMP_FREQ])
                if (
                        run_onset_speed_bef_range[0] <= temp_bef <= run_onset_speed_bef_range[1] and
                        run_onset_speed_aft_range[0] <= temp_aft <= run_onset_speed_aft_range[1]
                    ):
                    matched_bouts.append(i)  # append this run-bout index 
                    run_bout_speeds_matched.append(curr_speed)
                    sess_run_bout_speeds_matched.append(curr_speed)
            
            # plot speed-matched curves 
            if len(matched_bouts) > 0:
                mean_sess_run_onset_speeds = smooth_convolve(
                    np.mean(
                        sess_run_onset_speeds, axis=0
                        )[:2*SAMP_FREQ],
                    sigma=SAMP_FREQ/100
                    )
                
                mean_sess_run_bout_speeds = smooth_convolve(
                    np.mean(
                        sess_run_bout_speeds, axis=0
                        )[:2*SAMP_FREQ],
                    sigma=SAMP_FREQ/100
                    )
                
                mean_sess_run_bout_speeds_matched = smooth_convolve(
                    np.mean(
                        sess_run_bout_speeds_matched, axis=0
                        )[:2*SAMP_FREQ],
                    sigma=SAMP_FREQ/100
                    )
                
                fig, axs = plt.subplots(1,2, figsize=(4,1.8))
                
                axs[0].plot(XAXIS_SPEED, mean_sess_run_onset_speeds,
                            color='k')
                axs[0].plot(XAXIS_SPEED, mean_sess_run_bout_speeds,
                            color='green')
                axs[0].set_title('run-bouts not matched')

                axs[1].plot(XAXIS_SPEED, mean_sess_run_onset_speeds,
                            color='k')
                axs[1].plot(XAXIS_SPEED, mean_sess_run_bout_speeds_matched,
                            color='red')
                axs[1].set_title('run-bouts matched')
                
                for i in range(2):
                    axs[i].set(xlabel='time from run-onset (s)',
                               ylabel='speed (cm/s)')
                    for s in ['top', 'right']:
                        axs[i].spines[s].set_visible(False)
                
                for ext in ['.png', '.pdf']:
                    fig.savefig(
                        r'Z:\Dinghao\code_dinghao\behaviour\LC_run_onset_run_bout'
                        rf'\{recname}_matched_bouts{ext}',
                        dpi=300,
                        bbox_inches='tight'
                        )
                
                plt.show()
                plt.close(fig)
            
            print(f'speed-matching done; {len(matched_bouts)} run-bouts matched')
        
        
        #### process single cells ####
        curr_trains = all_trains[cluname]
        curr_trains_good = curr_trains[good_idx]
        
        run_onset_length = 7 * SAMP_FREQ
        run_onset_all = [
            np.pad(
                trial[:run_onset_length], 
                (0, max(0, run_onset_length - len(trial))), 
                constant_values=0
                ) 
            for trial in curr_trains_good
            ]
        run_onset_mean = np.mean(run_onset_all, axis=0)
        # run_onset_sem = sem(run_onset_all, axis=0)
        
        # signal to noise calculation
        run_onset_peak = max(
            run_onset_mean[
                int(RUN_ONSET_BIN - .25 * SAMP_FREQ) : int(RUN_ONSET_BIN + .25 * SAMP_FREQ)
                ]
            )  # peak
        run_onset_baseline = (
            np.mean(
                run_onset_mean[
                    int(RUN_ONSET_BIN - 1 * SAMP_FREQ) : int(RUN_ONSET_BIN - .5 * SAMP_FREQ)
                    ]
                ) + 
            np.mean(
                run_onset_mean[
                    int(RUN_ONSET_BIN + .5 * SAMP_FREQ) : int(RUN_ONSET_BIN + 1 * SAMP_FREQ)
                    ]
                )
            ) / 2  # baseline is the mean of -1~-.5 plus .5~1
    
        # times = run_bout_file['timeStepRun']
        fsa = run_bout_file['filteredSpikeArrayRunBoutOnSet'][clunum]  # bout x time
        
        #### if enough matched bouts, append data #### 
        # if fsa.shape[0]==9201 or len(matched_bouts)<3:  # to prevent contamination
        if fsa.shape[0]==9201 or len(matched_bouts)<3:
            continue
        else:
            fsa_mean = np.nanmean(fsa[matched_bouts, : 7 * SAMP_FREQ_RUNBOUT], 
                                  axis=0)
            # fsa_sem = sem(fsa[matched_bouts, : 7 * SAMP_FREQ_RUNBOUT], 
            #               axis=0)
            mean_run_onset.append(run_onset_mean)
            mean_run_bout.append(fsa_mean)
        
        run_bout_peak = max(
            fsa_mean[
                int(RUN_ONSET_BIN_RUNBOUT - .25 * SAMP_FREQ_RUNBOUT) : int(RUN_ONSET_BIN_RUNBOUT + .25 * SAMP_FREQ_RUNBOUT)])
        run_bout_baseline = (
            np.mean(
                fsa_mean[
                    int(RUN_ONSET_BIN_RUNBOUT - 1 * SAMP_FREQ_RUNBOUT) : int(RUN_ONSET_BIN_RUNBOUT - .5 * SAMP_FREQ_RUNBOUT)
                    ]
                ) + 
            np.mean(
                fsa_mean[
                    int(RUN_ONSET_BIN_RUNBOUT +.5 * SAMP_FREQ_RUNBOUT) : int(RUN_ONSET_BIN_RUNBOUT + 1 * SAMP_FREQ_RUNBOUT)
                    ]
                )
            ) / 2
        
        RO_run_bout.append(
            np.mean(
                fsa_mean[
                    int(RUN_ONSET_BIN_RUNBOUT - .25 * SAMP_FREQ_RUNBOUT) : int(RUN_ONSET_BIN_RUNBOUT + .25 * SAMP_FREQ_RUNBOUT)
                    ]
                )
            )
        s2n_run_bout.append(
            np.mean(
                fsa_mean[
                    int(RUN_ONSET_BIN_RUNBOUT - .25 * SAMP_FREQ_RUNBOUT) : int(RUN_ONSET_BIN_RUNBOUT + .25 * SAMP_FREQ_RUNBOUT)
                    ]
                ) / run_bout_baseline
            )
        peak_run_bout.append(run_bout_peak)
        
        RO_run_onset.append(
            np.mean(
                run_onset_mean[
                    int(RUN_ONSET_BIN - .25 * SAMP_FREQ) : int(RUN_ONSET_BIN + .25 * SAMP_FREQ)
                    ]
                )
            )
        s2n_run_onset.append(
            np.mean(
                run_onset_mean[
                    int(RUN_ONSET_BIN - .25 * SAMP_FREQ) : int(RUN_ONSET_BIN + .25 * SAMP_FREQ)
                    ]
                ) / run_onset_baseline
            )
        peak_run_onset.append(run_onset_peak)
    
    
        #### plotting ####
        fig, ax = plt.subplots(figsize=(3,2))
        
        ax.plot(
            run_onset_xaxis, 
            run_onset_mean[
                RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ
                ], 
            color='navy'
            )
        ax.plot(
            run_bout_xaxis, 
            fsa_mean[
                RUN_ONSET_BIN_RUNBOUT - BEF * SAMP_FREQ_RUNBOUT : RUN_ONSET_BIN_RUNBOUT + AFT * SAMP_FREQ_RUNBOUT], 
            color='gainsboro'
            )
        
        ax.set(title=cluname)
        
        out_dir = r'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\single_cell'
        fig.savefig(rf'{out_dir}\{cluname}.png',
                    dpi=300,
                    bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
        
    
    #### process accumulated session data ####
    # we need to filter out the inf's first 
    s2n_run_onset, s2n_run_bout = zip(*[
        (ro, rb) for ro, rb 
        in zip(s2n_run_onset, s2n_run_bout) 
        if np.isfinite(ro) and np.isfinite(rb)
    ])  # this gives tuples though, so keep that in mind
    
    all_sem_run_onset = sem(mean_run_onset, axis=0)
    all_mean_run_onset = np.nanmean(mean_run_onset, axis=0)
    all_sem_run_bout = sem(mean_run_bout, axis=0)
    all_mean_run_bout = np.nanmean(mean_run_bout, axis=0)
    
    # plot mean profiles
    wilc_res = wilcoxon(RO_run_bout, RO_run_onset)[1]
    
    print('plotting mean spiking profiles...')
    
    fig, ax = plt.subplots(figsize=(2,1.4))
    
    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)
    
    mean_run_onset_plot = all_mean_run_onset[
        RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ
        ]
    sem_run_onset_plot = all_sem_run_onset[
        RUN_ONSET_BIN - BEF * SAMP_FREQ : RUN_ONSET_BIN + AFT * SAMP_FREQ
        ]
    mean_run_bout_plot = all_mean_run_bout[
        RUN_ONSET_BIN_RUNBOUT - BEF * SAMP_FREQ_RUNBOUT : RUN_ONSET_BIN_RUNBOUT + AFT * SAMP_FREQ_RUNBOUT
        ]
    sem_run_bout_plot = all_sem_run_bout[
        RUN_ONSET_BIN_RUNBOUT - BEF * SAMP_FREQ_RUNBOUT : RUN_ONSET_BIN_RUNBOUT + AFT * SAMP_FREQ_RUNBOUT
        ]
    
    # xmax = max(max(mean_run_onset_plot), max(mean_run_bout_plot))
    # xmin = min(min(mean_run_onset_plot), min(mean_run_bout_plot))
    
    mean_run_onset_ln, = ax.plot(run_onset_xaxis, mean_run_onset_plot, 
                                 color='navy', zorder=10)
    mean_run_bout_ln, = ax.plot(run_bout_xaxis, mean_run_bout_plot, 
                                color='grey')
    ax.fill_between(run_onset_xaxis,
                    mean_run_onset_plot+sem_run_onset_plot,
                    mean_run_onset_plot-sem_run_onset_plot,
                    color='royalblue',
                    alpha=.25, edgecolor='none', zorder=10)
    ax.fill_between(run_bout_xaxis,
                    mean_run_bout_plot+sem_run_bout_plot,
                    mean_run_bout_plot-sem_run_bout_plot,
                    color='gainsboro',
                    alpha=.25, edgecolor='none')
    ax.set(xlim=(-1,4), xticks=[0,2,4],
           ylim=(1.8,5.3), yticks=[2,4], 
           title=f'Run-onset v run-bout-onset\n({list_identity} Dbh+)',
           xlabel='Time from run onset (s)',
           ylabel='Spike rate (Hz)')
    ax.legend([mean_run_onset_ln, mean_run_bout_ln], 
              ['Trial run onset', 'Run-bout onset'], 
              frameon=False, fontsize=6)
    
    plt.plot([-.5,.5], [5.1,5.1], c='k', lw=.5)
    plt.text(0, 5.1, f'wilc={round(wilc_res, 8)}', 
             ha='center', va='bottom', color='k', fontsize=5)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout'
            rf'\LC_{list_identity}_Dbh_ROpeaking_run_onset_run_bout{ext}',
            dpi=300, 
            bbox_inches='tight'
            )
        
    plt.show()
    
    # statistics 
    pf.plot_violin_with_scatter(s2n_run_onset, s2n_run_bout, 
                                'navy', 'grey', 
                                paired=True, 
                                xticklabels=['run-onset', 'run-bout\nonset'], 
                                ylabel='run-onset SNR', 
                                title='LC SNR', 
                                save=True, 
                                savepath=rf'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\LC_{list_identity}_Dbh_ROpeaking_run_onset_run_bout_SNR_violin', 
                                dpi=300)
    
    pf.plot_violin_with_scatter(RO_run_onset, RO_run_bout, 
                                'navy', 'grey', 
                                paired=True, 
                                xticklabels=['run-onset', 'run-bout\nonset'], 
                                ylabel='spike rate (Hz)', 
                                title='LC run-onset', 
                                save=True, 
                                savepath=rf'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\LC_{list_identity}_Dbh_ROpeaking_run_onset_run_bout_amp_violin', 
                                dpi=300)
    
    
    # speed match plotting 
    print('plotting overall speed comparison...')

    run_onset_speeds_mat = np.stack(run_onset_speeds, axis=0)
    run_bout_speeds_mat = np.stack(run_bout_speeds, axis=0)
    run_bout_speeds_matched_mat = np.stack(run_bout_speeds_matched, axis=0)
    
    # compute mean and sem
    mean_run_onset_speed = smooth_convolve(np.mean(
        run_onset_speeds_mat, axis=0
        )[:2*SAMP_FREQ], 
        sigma=SAMP_FREQ/100)
    sem_run_onset_speed = smooth_convolve(sem(
        run_onset_speeds_mat, axis=0
            )[:2*SAMP_FREQ], 
        sigma=SAMP_FREQ/100)
    
    mean_run_bout_speed = smooth_convolve(np.mean(
        run_bout_speeds_mat, axis=0
        )[:2*SAMP_FREQ], 
        sigma=SAMP_FREQ/100)
    sem_run_bout_speed = smooth_convolve(sem(
        run_bout_speeds_mat, axis=0
        )[:2*SAMP_FREQ], 
        sigma=SAMP_FREQ/100)
    
    mean_run_bout_speed_matched = smooth_convolve(np.mean(
        run_bout_speeds_matched_mat, axis=0
        )[:2*SAMP_FREQ], 
        sigma=SAMP_FREQ/100)
    sem_run_bout_speed_matched = smooth_convolve(sem(
        run_bout_speeds_matched_mat, axis=0
        )[:2*SAMP_FREQ], 
        sigma=SAMP_FREQ/100)
    
    fig, axs = plt.subplots(1,2, figsize=(4,1.8))
    
    # panel 1: not matched
    axs[0].plot(XAXIS_SPEED, mean_run_onset_speed, color='k')
    axs[0].plot(XAXIS_SPEED, mean_run_bout_speed, color='green')
    axs[0].fill_between(XAXIS_SPEED,
                        mean_run_onset_speed + sem_run_onset_speed,
                        mean_run_onset_speed - sem_run_onset_speed,
                        color='k', alpha=.1, edgecolor='none')
    axs[0].fill_between(XAXIS_SPEED,
                        mean_run_bout_speed + sem_run_bout_speed,
                        mean_run_bout_speed - sem_run_bout_speed,
                        color='green', alpha=.1, edgecolor='none')
    axs[0].set(title='run-bouts not matched')
    
    # panel 2: matched
    axs[1].plot(XAXIS_SPEED, mean_run_onset_speed, color='k')
    axs[1].plot(XAXIS_SPEED, mean_run_bout_speed_matched, color='red')
    axs[1].fill_between(XAXIS_SPEED,
                        mean_run_onset_speed + sem_run_onset_speed,
                        mean_run_onset_speed - sem_run_onset_speed,
                        color='k', alpha=.1, edgecolor='none')
    axs[1].fill_between(XAXIS_SPEED,
                        mean_run_bout_speed_matched + sem_run_bout_speed_matched,
                        mean_run_bout_speed_matched - sem_run_bout_speed_matched,
                        color='red', alpha=.1, edgecolor='none')
    axs[1].set(title='run-bouts matched')
    
    for ax in axs:
        ax.set(xlim=(-1,1), xticks=[-1,0,1],
               ylim=(0, max(mean_run_onset_speed.max(), 
                            mean_run_bout_speed.max(), 
                            mean_run_bout_speed_matched.max()) * 1.1),
               xlabel='Time (s)',
               ylabel='Speed (cm/s)')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
    
    for ext in ['.png', '.pdf']:
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\behaviour\LC_run_onset_run_bout'
            rf'\overall_matched_bouts{ext}',
            dpi=300,
            bbox_inches='tight'
        )
    
    plt.show()
    plt.close(fig)
    
    
if __name__ == '__main__':
    main(tagged_RO_peak_keys, 'tagged')
    main(putative_RO_peak_keys, 'putative')