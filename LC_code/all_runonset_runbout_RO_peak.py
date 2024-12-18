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
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
RO_peaking_keys = []
for cell in cell_prop.index:
    up = cell_prop['peakness'][cell]  # union-peakness
    pt = cell_prop['putative'][cell]  # putative
    tg = cell_prop['tagged'][cell]
    
    if up and pt:
        RO_peaking_keys.append(cell)
    if up and tg:  # since putative does not include tagged
        RO_peaking_keys.append(cell)


#%% MAIN
all_ro_rb = {}

avg_ro = []; avg_rb = []
pk_ro = []; pk_rb = []
s2n_run_onset = []
s2n_run_bout = []

for clu in list(all_train.items()):
    cluname = clu[0]
    
    if cluname not in RO_peaking_keys:
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
    run_onset_mean = np.mean([trial[1250:8750]*1250 for trial in curr_spike_good if len(trial)>=8750], 
                             axis=0)
    run_onset_sem = sem([trial[1250:8750]*1250 for trial in curr_spike_good if len(trial)>=8750],
                        axis=0)
    bl_run_onset = (np.mean(run_onset_mean[625:1875])+np.mean(run_onset_mean[5625:6875]))/2
    s2n_run_onset.append(run_onset_mean[3750]/bl_run_onset)
    
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
    
    if fsa.shape[0]==9201 or len(matched_bouts)<=10:
        pass
    else:
        fsa_mean = np.mean(fsa[matched_bouts, 400:2800], axis=0)  # 2s around bout-onset
        fsa_sem = sem(fsa[matched_bouts, 400:2800], axis=0)
        all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
        avg_ro.append(run_onset_mean)
        avg_rb.append(fsa_mean)
        # pk_ro.append(np.mean(run_onset_mean[1875:3125]))
        # pk_rb.append(np.mean(fsa_mean[600:1000]))
        pk_ro.append(run_onset_mean[2500])
        pk_rb.append(fsa_mean[800])
    
    bl_run_bout = (np.mean(fsa_mean[200:600])+np.mean(fsa_mean[1000:1400]))/2
    s2n_run_bout.append(fsa_mean[800]/bl_run_bout)


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
w_res = wilcoxon(pk_ro, pk_rb)[1]

print('plotting average spiking profiles...')
fig, ax = plt.subplots(figsize=(2,1.4))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

avg_ro_plot = avg_ro[1250:7500]
sem_ro_plot = sem_ro[1250:7500]
avg_rb_plot = avg_rb[400:2400]
sem_rb_plot = sem_rb[400:2400]

avg_ro_ln, = ax.plot(ro_xaxis, avg_ro_plot, color='navy')
avg_rb_ln, = ax.plot(rb_xaxis, avg_rb_plot, color='grey')
ax.fill_between(ro_xaxis,
                avg_ro_plot+sem_ro_plot,
                avg_ro_plot-sem_ro_plot,
                color='royalblue',
                alpha=.1, edgecolor='none')
ax.fill_between(rb_xaxis,
                avg_rb_plot+sem_rb_plot,
                avg_rb_plot-sem_rb_plot,
                color='gainsboro',
                alpha=.1, edgecolor='none')
# ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(xlim=(-1,4), xticks=[0,2,4],
       ylim=(1.5,8.5), yticks=[3,5,7], 
       title='RO v Rb-onset (all Dbh+)',
       xlabel='time (s)',
       ylabel='spike rate (Hz)')
ax.legend([avg_ro_ln, avg_rb_ln], ['trial run onset', 'run-bout onset'], frameon=False, fontsize=8)

plt.plot([-.5,.5], [8.05,8.05], c='k', lw=.5)
plt.text(0, 8.05, 'p={}'.format(round(w_res, 8)), ha='center', va='bottom', color='k', fontsize=5)

plt.show()

for ext in ['.png', '.pdf']:
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_allDbh_ROpeaking{}'.format(ext),
                dpi=300, bbox_inches='tight')


#%% statistics 
pf.plot_violin_with_scatter(s2n_run_onset, s2n_run_bout, 
                            'navy', 'grey', 
                            paired=True, 
                            xticklabels=['run-onset', 'run-bout\nonset'], 
                            ylabel='run-onset burst SNR', 
                            title='LC RO-SNR', 
                            save=True, savepath=r'Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_allDbh_ROpeaking_violin', 
                            dpi=300)