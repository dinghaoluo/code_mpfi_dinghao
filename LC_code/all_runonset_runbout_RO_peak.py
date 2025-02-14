# -*- coding: utf-8 -*-
"""
Created on Sun June 11 13:12:54 2023

LC: visual and statistical comparison between run-onset and run-bout-onset LC
    burst amplitudes

*use putative Dbh RO peaking cells*

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

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()
import plotting_functions as pf

sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% load data 
all_train = np.load(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_trains.npy',
    allow_pickle=True
    ).item()
cell_prop = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )


#%% specify RO peaking putative Dbh cells
all_clus = []
for cell in cell_prop.index:
    run_on_burst = cell_prop['run_onset_peak'][cell]
    identity = cell_prop['identity'][cell]
    
    if run_on_burst and identity!='other':
        all_clus.append(cell)


#%% MAIN
all_ro_rb = {}

avg_ro = []; avg_rb = []

s2n_run_onset = []
s2n_run_bout = []

peak_run_onset = []
peak_run_bout = []

for clu in list(all_train.items()):
    cluname = clu[0]
    
    if cluname not in all_clus:
        continue
    
    print('processing {}'.format(cluname))
    clunum = int(cluname[21:])-2  # index for retrieving fsa
    pathname = 'Z:\Dinghao\MiceExp\ANMD'+cluname[1:5]+'\\'+cluname[:14]+'\\'+cluname[:17]
    
    # load bad trial indices
    behPar = sio.loadmat(pathname+pathname[42:]+
                         '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                         # -1 to account for MATLAB Python difference
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
    
    curr_spike_all = clu[1]
    curr_spike_good = curr_spike_all[ind_good_beh]
    run_onset_mean = np.mean([trial[:8750]*1250 for trial in curr_spike_good if len(trial)>=8750], 
                             axis=0)
    run_onset_sem = sem([trial[:8750]*1250 for trial in curr_spike_good if len(trial)>=8750],
                        axis=0)
    
    # signal to noise calculation
    run_onset_peak = max(run_onset_mean[int(3750-.25*1250):int(3750+.25*1250)])  # peak
    run_onset_baseline = np.mean(run_onset_mean[int(3750+1250*.75):int(3750+1250*1.25)])  # .75~1.25 s after RO
    
    peak_run_onset.append(run_onset_peak)
    s2n_run_onset.append(run_onset_peak/run_onset_baseline)
    
    # for speed matching
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    speed_f = sio.loadmat(filename + '_alignRun_msess1.mat')
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

    ro_speed_bef_mean = np.mean(ro_speed_bef, axis=0)
    ro_speed_bef_std = np.std(ro_speed_bef, axis=0)
    ro_speed_aft_mean = np.mean(ro_speed_aft, axis=0)
    ro_speed_aft_std = np.std(ro_speed_aft, axis=0)
    ro_speed_bef_range = [ro_speed_bef_mean-ro_speed_bef_std,
                          ro_speed_bef_mean+ro_speed_bef_std]
    ro_speed_aft_range = [ro_speed_aft_mean-ro_speed_aft_std,
                          ro_speed_aft_mean+ro_speed_aft_std]
    
    # filename = input('file route: ')
    filestem = pathname+pathname[-18:]
    
    # import beh file
    run_bout_file_name = r'Z:\Dinghao\code_dinghao\run_bouts\fsa_run_bouts'+pathname[-18:]+'_BefRunBout0.mat'
    run_bout_table_name = r'Z:\Dinghao\code_dinghao\run_bouts'+pathname[-18:]+'_run_bouts_py.csv'
    run_bout_file = mat73.loadmat(run_bout_file_name)
    run_bout_table = pd.read_csv(run_bout_table_name)
    run_bout_starts = list(run_bout_table.iloc[:,1])
    speed_all = mat73.loadmat(filestem+'_BehavElectrDataLFP.mat')['Track']['speed_MMsec']

    matched_bouts = []
    for i in range(len(run_bout_starts)):
        start = run_bout_starts[i]
        rbsbef = np.mean(speed_all[start-1875:start])
        rbsaft = np.mean(speed_all[start:start+1875])
        if rbsaft>=ro_speed_aft_range[0] and rbsaft<=ro_speed_aft_range[1]:
            matched_bouts.append(i)

    times = run_bout_file['timeStepRun']
    fsa = run_bout_file['filteredSpikeArrayRunBoutOnSet'][clunum]  # bout x time
    
    if fsa.shape[0]==9201 or len(matched_bouts)<=10:  # to prevent contamination
        pass
    else:
        fsa_mean = np.mean(fsa[matched_bouts, :2800], axis=0)
        fsa_sem = sem(fsa[matched_bouts, :2800], axis=0)
        all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
        avg_ro.append(run_onset_mean)
        avg_rb.append(fsa_mean)
    
    run_bout_peak = max(fsa_mean[int(1200-400*.25):int(1200+400*.25)])
    run_bout_baseline = np.mean(fsa_mean[int(1200+400*.75):int(1200+400*1.25)])
    
    peak_run_bout.append(run_bout_peak)
    s2n_run_bout.append(run_bout_peak/run_bout_baseline)


#%% we need to filter out the inf's first 
s2n_run_onset, s2n_run_bout = zip(*[
    (ro, rb) for ro, rb in zip(s2n_run_onset, s2n_run_bout) if np.isfinite(ro) and np.isfinite(rb)
])  # this gives tuples though, so keep that in mind


#%% speed matching finished
print('speed matching finished')

sem_ro = sem(avg_ro, axis=0)
avg_ro = np.mean(avg_ro, axis=0)
sem_rb = sem(avg_rb, axis=0)
avg_rb = np.mean(avg_rb, axis=0)


# %% plotting all
# print('plotting all spiking profiles...')
# tot_plots = len(all_ro_rb)
# col_plots = 5
# row_plots = tot_plots // col_plots
# if tot_plots % col_plots != 0:
#     row_plots += 1
# plot_pos = np.arange(1, tot_plots+1)

# fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()

rb_xaxis = np.arange(-400, 1600)*.0025
ro_xaxis = np.arange(-1250, 5000)/1250

# i = 0
# for clu in list(all_ro_rb.items()):
#     ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
        
#     cluname = clu[0]
#     ro_curr = all_ro_rb[cluname][0][1250:7500]  # -1~4s
#     ro_sem_curr = all_ro_rb[cluname][1][1250:7500]  # -1~4s
#     rb_curr = all_ro_rb[cluname][2][400:2400]  #-1~4s
#     rb_sem_curr = all_ro_rb[cluname][3][400:2400]  #-1~4s
#     ro_ln, = ax.plot(ro_xaxis, ro_curr)
#     rb_ln, = ax.plot(rb_xaxis, rb_curr)
#     ax.fill_between(ro_xaxis, 
#                     ro_curr+ro_sem_curr,
#                     ro_curr-ro_sem_curr, 
#                     alpha=.1)
#     ax.fill_between(rb_xaxis, 
#                     rb_curr+rb_sem_curr,
#                     rb_curr-rb_sem_curr, 
#                     alpha=.1)
#     max_curr = max([max(ro_curr), max(rb_curr)])
#     ax.set(ylim=(0,max_curr*1.5),
#            xlim=(-1,4),
#            title=cluname, 
#            xlabel='time (s)',
#            ylabel='spike rate (Hz)')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.vlines(0, 0, 20, color='grey', alpha=.1)
#     ax.legend([ro_ln, rb_ln], ['run onset', 'run bout'])
    
#     i+=1

# plt.subplots_adjust(hspace = 0.5)
# plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_putDbh_ROpeaking.png',
#             dpi=300,
#             bbox_inches='tight')


#%% plotting average
w_res = wilcoxon(peak_run_bout, peak_run_onset)[1]

print('plotting average spiking profiles...')
fig, ax = plt.subplots(figsize=(2,1.4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

avg_ro_plot = avg_ro[3750-1250:3750+1250*4]
sem_ro_plot = sem_ro[3750-1250:3750+1250*4]
avg_rb_plot = avg_rb[1200-400:1200+400*4]
sem_rb_plot = sem_rb[1200-400:1200+400*4]

avg_ro_ln, = ax.plot(ro_xaxis, avg_ro_plot, color='navy', zorder=10)
avg_rb_ln, = ax.plot(rb_xaxis, avg_rb_plot, color='grey')
ax.fill_between(ro_xaxis,
                avg_ro_plot+sem_ro_plot,
                avg_ro_plot-sem_ro_plot,
                color='royalblue',
                alpha=.1, edgecolor='none', zorder=10)
ax.fill_between(rb_xaxis,
                avg_rb_plot+sem_rb_plot,
                avg_rb_plot-sem_rb_plot,
                color='gainsboro',
                alpha=.1, edgecolor='none')
# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(xlim=(-1,4), xticks=[0,2,4],
       ylim=(2.2,5.8), yticks=[3,5], 
       title='RO v Rb-onset (all Dbh+)',
       xlabel='time (s)',
       ylabel='spike rate (Hz)')
ax.legend([avg_ro_ln, avg_rb_ln], ['trial run onset', 'run-bout onset'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [5.55,5.55], c='k', lw=.5)
plt.text(0, 5.55, 'p={}'.format(round(w_res, 8)), ha='center', va='bottom', color='k', fontsize=5)

plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_rovrb_avg_allDbh_ROpeaking{}'.format(ext),
                dpi=300, bbox_inches='tight')


#%% statistics 
pf.plot_violin_with_scatter(s2n_run_onset, s2n_run_bout, 
                            'navy', 'grey', 
                            paired=True, 
                            xticklabels=['run-onset', 'run-bout\nonset'], 
                            ylabel='run-onset burst SNR', 
                            title='LC SNR', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_rovrb_avg_allDbh_ROpeaking_violin', 
                            dpi=300)