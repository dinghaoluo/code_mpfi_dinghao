# -*- coding: utf-8 -*-
"""
Created on Fri 3 Feb 2023

**GENERAL POPULATION**
plot run-bouts identified using 'Z:\Dinghao\code\runBouts' (modified Raphi's)

analyse run-onset vs run-bout-onset burst with peaking neurons

updated 20 Feb 2023 for speed-matching between trial onsets and run-bout-onsets

@author: Dinghao Luo
"""


#%% imports
import numpy as np
from scipy.stats import sem, ttest_rel, wilcoxon
import scipy.io as sio
import matplotlib.pyplot as plt
import mat73
import pandas as pd


#%% MAIN
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()
all_ro_rb = {}
clstr = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_clustered_hierarchical_centroid.npy',
                allow_pickle=True).item()
clstr_clu = list(clstr['clustering result'].values())
clstr_name = list(clstr['clustering result'].keys())
burst = []
others = []
for i in range(len(clstr['clustering result'])):
    if clstr_clu[i]=='3':
        burst.append(clstr_name[i])
    else:
        others.append(clstr_name[i])
avg_ro = []; avg_rb = []
burst_ro = []; burst_rb = []

for clu in list(all_train.items()):
    cluname = clu[0]
    print('processing {}'.format(cluname))
    clunum = int(cluname[21:])-2  # index for retrieving fsa
    pathname = 'Z:\Dinghao\MiceExp\ANMD'+cluname[1:5]+'\\'+cluname[:14]+'\\'+cluname[:17]
    
    # load bad trial indices
    beh_par_file = sio.loadmat(pathname+pathname[42:]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    curr_spike_all = clu[1]
    curr_spike_good = curr_spike_all[ind_good_beh]
    run_onset_mean = np.mean([trial[1250:8750]*1250 for trial in curr_spike_good if len(trial)>=8750], 
                             axis=0)
    run_onset_sem = sem([trial[1250:8750]*1250 for trial in curr_spike_good if len(trial)>=8750],
                        axis=0)
    
    # for speed matching
    filename = pathname + pathname[-18:] + '_DataStructure_mazeSection1_TrialType1'
    speed_f = sio.loadmat(filename + '_alignRun_msess1.mat')
    ro_speed_bef = []; ro_speed_aft = []
    for ind in ind_good_beh:
        bef_curr = speed_f['trialsRun'][0][0]['speed_MMsecBef'][0][ind+1][1875:]
        for tbin in range(1875):
            if bef_curr[tbin] < 0:
                if tbin == 1874:
                    bef_curr[tbin] = bef_curr[tbin-1]
                elif tbin == 0:
                    bef_curr[tbin] = bef_curr[tbin+1]
                else:
                    bef_curr[tbin] = (bef_curr[tbin-1]+bef_curr[tbin+1])/2
        ro_speed_bef.append(np.mean(bef_curr))
        ro_speed_aft.append(np.mean(speed_f['trialsRun'][0][0]['speed_MMsec'][0][ind+1][:1875]))

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
    
    # peak_det = np.mean([trial[:6250] for trial in curr_spike_all], axis=0)
    # if fsa.shape[0]==9201 or fsa.shape[0]<=50 or neu_peak_detection(peak_det)==False:
    #     pass
    if fsa.shape[0]==9201 or cluname in others or len(matched_bouts)<=15:
        pass
    else:
        fsa_mean = np.mean(fsa[matched_bouts, 400:2800], axis=0)  # 2s around bout-onset
        fsa_sem = sem(fsa[matched_bouts, 400:2800], axis=0)
        all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
        avg_ro.append(run_onset_mean)
        avg_rb.append(fsa_mean)
        burst_ro.append(np.mean(run_onset_mean[2250:2750]))  # .4s around onset
        burst_rb.append(np.mean(fsa_mean[640:960]))


#%% speed matching finished
print('speed matching finished')

sem_ro = sem(avg_ro, axis=0)
avg_ro = np.mean(avg_ro, axis=0)
sem_rb = sem(avg_rb, axis=0)
avg_rb = np.mean(avg_rb, axis=0)


#%% plotting all
print('plotting all spiking profiles...')
tot_plots = len(all_ro_rb)
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()

rb_xaxis = np.arange(-400, 1600)*.0025
ro_xaxis = np.arange(-1250, 5000)/1250

i = 0
for clu in list(all_ro_rb.items()):
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
        
    cluname = clu[0]
    ro_curr = all_ro_rb[cluname][0][1250:7500]  # -1~4s
    ro_sem_curr = all_ro_rb[cluname][1][1250:7500]  # -1~4s
    rb_curr = all_ro_rb[cluname][2][400:2400]  #-1~4s
    rb_sem_curr = all_ro_rb[cluname][3][400:2400]  #-1~4s
    ro_ln, = ax.plot(ro_xaxis, ro_curr)
    rb_ln, = ax.plot(rb_xaxis, rb_curr)
    ax.fill_between(ro_xaxis, 
                    ro_curr+ro_sem_curr,
                    ro_curr-ro_sem_curr, 
                    alpha=.1)
    ax.fill_between(rb_xaxis, 
                    rb_curr+rb_sem_curr,
                    rb_curr-rb_sem_curr, 
                    alpha=.1)
    max_curr = max([max(ro_curr), max(rb_curr)])
    ax.set(ylim=(0,max_curr*1.5),
           xlim=(-1,4),
           title=cluname, 
           xlabel='time (s)',
           ylabel='spike rate (Hz)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.vlines(0, 0, 20, color='grey', alpha=.1)
    ax.legend([ro_ln, rb_ln], ['run onset', 'run bout'])
    
    i+=1

plt.subplots_adjust(hspace = 0.5)
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_clstr1.png',
            dpi=300,
            bbox_inches='tight')


#%% plotting average
print('plotting average spiking profiles...')
fig, ax = plt.subplots(figsize=(5,4))
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
                alpha=.1)
ax.fill_between(rb_xaxis,
                avg_rb_plot+sem_rb_plot,
                avg_rb_plot-sem_rb_plot,
                color='gainsboro',
                alpha=.1)
ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(xlim=(-1,4),
       ylim=(1.5,6),
       title='trial run onset v run-bout onset (cluster 1)',
       xlabel='time (s)',
       ylabel='spike rate (Hz)')
ax.legend([avg_ro_ln, avg_rb_ln], ['trial run onset', 'run-bout onset'])

plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_clstr1.png',
            dpi=300,
            bbox_inches='tight')


#%% t-test and plotting bar graph
t_res = ttest_rel(a=burst_ro, b=burst_rb)
pval = t_res[1]
print('t_test: {}'.format(t_res))

wilc = wilcoxon(burst_ro, burst_rb, alternative='two-sided')
pvalwilc = wilc[1]
print('Wilcoxon: {}'.format(wilc))

fig, ax = plt.subplots(figsize=(4,4))

x = [0, 20]; y = [0, 20]
ax.plot(x, y, color='grey')
ax.scatter(burst_ro, burst_rb, s=5, color='grey', alpha=.5)
mean_bro = np.mean(burst_ro); mean_brb = np.mean(burst_rb)
sem_bro = sem(burst_ro); sem_brb = sem(burst_ro)
ax.scatter(mean_bro, mean_brb, s=15, color='navy', alpha=.9)
ax.plot([mean_bro, mean_bro], 
        [mean_brb+sem_brb, mean_brb-sem_brb], 
        color='royalblue', alpha=.7)
ax.plot([mean_bro+sem_bro, mean_bro-sem_bro], 
        [mean_brb, mean_brb],
        color='royalblue', alpha=.7)

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc),
       xlabel='trial run onset',
       ylabel='run-bout onset',
       xlim=(0,18), ylim=(0,18))

plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_clstr1_bivariate.png',
            dpi=300,
            bbox_inches='tight')


fig, ax = plt.subplots(figsize=(3,6))

xaxis = [1, 2]
ax.bar(xaxis, 
       [np.mean(burst_ro), np.mean(burst_rb)],
       yerr=[sem(burst_ro), sem(burst_rb)], capsize=5,
       width=0.8,
       tick_label=['run-onset', 'run-bout'],
       edgecolor=['royalblue', 'grey'],
       color=(0,0,0,0))

ax.scatter(1+np.random.random(len(burst_ro))*0.5-0.25, burst_ro,
           s=3, color='royalblue', alpha=.5)
ax.scatter(2+np.random.random(len(burst_rb))*0.5-0.25, burst_rb,
           s=3, color='grey', alpha=.5)

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc),
       ylabel='spike rate (Hz)')

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_clstr1_bar.png',
            dpi=300,
            bbox_inches='tight')


#%% plot speed-matched examples 
# print('plotting speed-matched example trials...')

# fig, [ax0, ax1] = plt.subplots(1, 2)

# gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
# sigma_speed = 1250/100  # samp_freq/100
# gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
#               np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]

# ro_eg = np.convolve(ro_eg, gaus_speed)
# rb_eg = np.convolve(rb_eg, gaus_speed)
# ax0.plot(ro_eg)
# ax1.plot(rb_eg)