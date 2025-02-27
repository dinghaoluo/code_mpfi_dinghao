# -*- coding: utf-8 -*-
"""
Created on Sun June 11 13:12:54 2023
Modified on Wedj 26 Feb 2025 to improve stability, fix bugs and install wrapper 

LC: visual and statistical comparison between run-onset and run-bout-onset LC
    burst amplitudes

@author: Dinghao Luo
"""

#%% imports
import sys
import numpy as np
import pandas as pd
from scipy.stats import sem, wilcoxon
import matplotlib.pyplot as plt 
import scipy.io as sio
import mat73
import os 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()
import plotting_functions as pf

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% load data 
all_trains = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_trains.npy',
    allow_pickle=True
    ).item()
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
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
    print(f'processing {list_identity}...\n')
    
    all_ro_rb = {}
    
    mean_run_onset = []
    mean_run_bout = []
    
    s2n_run_onset = []
    s2n_run_bout = []
    
    peak_run_onset = []
    peak_run_bout = []
    
    RO_run_onset = []
    RO_run_bout = []
    
    run_onset_xaxis = np.arange(-1250, 5000)/1250
    run_bout_xaxis = np.arange(-400, 1600)*.0025
    
    sessname = ''
    
    for cluname in keys:
        if 'A060r-20230602-01' in cluname:
            continue
        
        # print('processing {}'.format(cluname))
        clunum = int(cluname[21:])-2  # index for retrieving fsa
        
        if cluname[:cluname.find(' ')] != sessname:
            sessname = cluname[:cluname.find(' ')]
            pathname = rf'Z:\Dinghao\MiceExp\ANMD{cluname[1:5]}\{cluname[:14]}\{cluname[:17]}'
        
            # load bad trial indices
            behPar = sio.loadmat(
                rf'{pathname}{pathname[42:]}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'
                )  # -1 to account for MATLAB Python difference
            ind_bad_beh = np.where(behPar['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                 # -1 to account for 0 being an empty trial
            ind_good_beh = np.arange(behPar['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
            ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
        
            # import stim trial indices
            stimOn = behPar['behPar']['stimOn'][0][0][0]
            first_stim = next((i for i, j in enumerate(stimOn) if j), None)
            if type(first_stim)==int:  # only the baseline trials
                ind_bad_beh = ind_bad_beh[ind_bad_beh<first_stim]
                ind_good_beh = ind_good_beh[ind_good_beh<first_stim]
                
            speed_f = sio.loadmat(
                rf'{pathname}{pathname[-18:]}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'
                )
        
        curr_spike_all = all_trains[cluname]
        curr_spike_good = curr_spike_all[ind_good_beh]
        
        run_onset_mean = np.mean([trial[:8750]*1250 for trial in curr_spike_good 
                                  if len(trial)>=8750], 
                                 axis=0)
        run_onset_sem = sem([trial[:8750]*1250 for trial in curr_spike_good 
                             if len(trial)>=8750],
                            axis=0)
        
        # signal to noise calculation
        run_onset_peak = max(run_onset_mean[int(3750-.25*1250):int(3750+.25*1250)])  # peak
        run_onset_baseline = (np.mean(run_onset_mean[int(3750-1250*1):int(3750-1250*.5)] + np.mean(run_onset_mean[int(3750+1250*.5):int(3750+1250*1)]))) / 2
        
        # speed matching
        ro_speed_bef = []; ro_speed_aft = []
        for ind in ind_good_beh:
            bef_curr = speed_f['trialsRun'][0][0]['speed_MMsecBef'][0][ind+1][1250:]
            for tbin in range(1250):
                if bef_curr[tbin] < 0:
                    if tbin == 1249:
                        bef_curr[tbin] = bef_curr[tbin-1]
                    elif tbin == 0:
                        bef_curr[tbin] = bef_curr[tbin+1]
                    else:
                        bef_curr[tbin] = (bef_curr[tbin-1]+bef_curr[tbin+1])/2
            ro_speed_bef.append(np.mean(bef_curr))
            ro_speed_aft.append(np.mean(speed_f['trialsRun'][0][0]['speed_MMsec'][0][ind+1][:1250]))
    
        # ro_speed_bef_mean = np.mean(ro_speed_bef, axis=0)
        # ro_speed_bef_std = np.std(ro_speed_bef, axis=0)
        ro_speed_aft_mean = np.mean(ro_speed_aft, axis=0)
        ro_speed_aft_std = np.std(ro_speed_aft, axis=0)
        # ro_speed_bef_range = [ro_speed_bef_mean-ro_speed_bef_std,
        #                       ro_speed_bef_mean+ro_speed_bef_std]
        ro_speed_aft_range = [ro_speed_aft_mean-ro_speed_aft_std,
                              ro_speed_aft_mean+ro_speed_aft_std]
        
        # import beh file
        run_bout_file = mat73.loadmat(
            rf'Z:\Dinghao\code_dinghao\run_bouts\fsa_run_bouts\{pathname[-18:]}_BefRunBout0.mat'
            )
        run_bout_table = pd.read_csv(
            rf'Z:\Dinghao\code_dinghao\run_bouts\{pathname[-18:]}_run_bouts_py.csv'
            )
        run_bout_starts = list(run_bout_table.iloc[:,1])
        speed_all = mat73.loadmat(
            rf'{pathname}{pathname[-18:]}_BehavElectrDataLFP.mat'
            )['Track']['speed_MMsec']
    
        matched_bouts = []
        for i in range(len(run_bout_starts)):
            start = run_bout_starts[i]
            # rbsbef = np.mean(speed_all[start-1875:start])
            rbsaft = np.mean(speed_all[start:start+1875])
            if rbsaft>=ro_speed_aft_range[0] and rbsaft<=ro_speed_aft_range[1]:
                matched_bouts.append(i)
    
        # times = run_bout_file['timeStepRun']
        fsa = run_bout_file['filteredSpikeArrayRunBoutOnSet'][clunum]  # bout x time
        
        if fsa.shape[0]==9201 or len(matched_bouts)<=3:  # to prevent contamination
            continue
        else:
            fsa_mean = np.mean(fsa[matched_bouts, :2800], axis=0)
            fsa_sem = sem(fsa[matched_bouts, :2800], axis=0)
            all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
            mean_run_onset.append(run_onset_mean)
            mean_run_bout.append(fsa_mean)
        
        run_bout_peak = max(fsa_mean[int(1200-400*.25):int(1200+400*.25)])
        run_bout_baseline = (np.mean(fsa_mean[int(1200-400*1):int(1200-400*.5)] + np.mean(fsa_mean[int(1200+400*.5):int(1200+400*1)]))) / 2
        
        RO_run_bout.append(np.mean(fsa_mean[int(1200-400*.25):int(1200+400*.25)]))
        peak_run_bout.append(run_bout_peak)
        s2n_run_bout.append(np.mean(fsa_mean[int(1200-400*.25):int(1200+400*.25)])/run_bout_baseline)
        
        RO_run_onset.append(np.mean(run_onset_mean[int(3750-1250*.25):int(3750+1250*.25)]))
        peak_run_onset.append(run_onset_peak)
        s2n_run_onset.append(np.mean(run_onset_mean[int(3750-1250*.25):int(3750+1250*.25)])/run_onset_baseline)
    
        ## plotting 
        fig, ax = plt.subplots(figsize=(3,2))
        
        ax.plot(run_onset_xaxis, run_onset_mean[2500:3750+4*1250], color='navy')
        ax.plot(run_bout_xaxis, fsa_mean[800:1200+400*4], color='gainsboro')
        
        ax.set(title=cluname)
        
        os.makedirs(r'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\single_cell',
                    exist_ok=True)
        fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\single_cell\{cluname}.png',
                    dpi=300,
                    bbox_inches='tight')
        
        plt.close(fig)
    
        print(f'processed {cluname}')
    
    # we need to filter out the inf's first 
    s2n_run_onset, s2n_run_bout = zip(*[
        (ro, rb) for ro, rb in zip(s2n_run_onset, s2n_run_bout) if np.isfinite(ro) and np.isfinite(rb)
    ])  # this gives tuples though, so keep that in mind
    
    
    ## speed matching finished
    print('speed matching finished')
    
    all_sem_run_onset = sem(mean_run_onset, axis=0)
    all_mean_run_onset = np.mean(mean_run_onset, axis=0)
    all_sem_run_bout = sem(mean_run_bout, axis=0)
    all_mean_run_bout = np.mean(mean_run_bout, axis=0)
    
    
    ## plotting mean profiles
    wilc_res = wilcoxon(RO_run_bout, RO_run_onset)[1]
    
    print('plotting mean spiking profiles...')
    
    fig, ax = plt.subplots(figsize=(2,1.4))
    
    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)
    
    mean_run_onset_plot = all_mean_run_onset[3750-1250:3750+1250*4]
    sem_run_onset_plot = all_sem_run_onset[3750-1250:3750+1250*4]
    mean_run_bout_plot = all_mean_run_bout[1200-400:1200+400*4]
    sem_run_bout_plot = all_sem_run_bout[1200-400:1200+400*4]
    
    mean_run_onset_ln, = ax.plot(run_onset_xaxis, mean_run_onset_plot, 
                                 color='navy', zorder=10)
    mean_run_bout_ln, = ax.plot(run_bout_xaxis, mean_run_bout_plot, 
                                color='grey')
    ax.fill_between(run_onset_xaxis,
                    mean_run_onset_plot+sem_run_onset_plot,
                    mean_run_onset_plot-sem_run_onset_plot,
                    color='royalblue',
                    alpha=.1, edgecolor='none', zorder=10)
    ax.fill_between(run_bout_xaxis,
                    mean_run_bout_plot+sem_run_bout_plot,
                    mean_run_bout_plot-sem_run_bout_plot,
                    color='gainsboro',
                    alpha=.1, edgecolor='none')
    ax.set(xlim=(-1,4), xticks=[0,2,4],
           ylim=(2.15,5.8), yticks=[3,5], 
           title=f'run-onset v run-bout-onset\n({list_identity} Dbh+)',
           xlabel='time (s)',
           ylabel='spike rate (Hz)')
    ax.legend([mean_run_onset_ln, mean_run_bout_ln], 
              ['trial run onset', 'run-bout onset'], 
              frameon=False, fontsize=6)
    
    plt.plot([-.5,.5], [5.55,5.55], c='k', lw=.5)
    plt.text(0, 5.55, 'wilc={}'.format(round(wilc_res, 8)), ha='center', va='bottom', color='k', fontsize=5)
    
    plt.show()
    
    os.makedirs(r'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout',
                exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(rf'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\LC_{list_identity}_Dbh_ROpeaking_run_onset_run_bout{ext}',
                    dpi=300, bbox_inches='tight')
    
    # statistics 
    pf.plot_violin_with_scatter(s2n_run_onset, s2n_run_bout, 
                                'navy', 'grey', 
                                paired=True, 
                                xticklabels=['run-onset', 'run-bout\nonset'], 
                                ylabel='run-onset burst SNR', 
                                title='LC SNR', 
                                save=True, 
                                savepath=rf'Z:\Dinghao\code_dinghao\LC_ephys\run_onset_v_run_bout\LC_{list_identity}_Dbh_ROpeaking_run_onset_run_bout_violin', 
                                dpi=300)
    
if __name__ == '__main__':
    main(tagged_RO_peak_keys, 'tagged')
    main(putative_RO_peak_keys, 'putative')