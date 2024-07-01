# -*- coding: utf-8 -*-
"""
Created on Sat 4 Feb 2023

plot run-bouts identified using 'Z:\Dinghao\code\runBouts' (modified Raphi's)

analyse run-onset vs run-bout-onset burst with trough neurons

@author: Dinghao Luo
"""


#%% imports
import numpy as np
from scipy.stats import sem, ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import mat73
import scipy.io as sio

#%% MAIN
all_tagged_train = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                           allow_pickle=True).item()
clustered = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_clustered.npy',
                    allow_pickle=True).item()
burstv = clustered['burstv']; burst = clustered['burst']; other = clustered['other']
all_ro_rb = {}
avg_ro = []; avg_rb = []
burst_ro = []; burst_rb = []

for clu in list(all_tagged_train.items()):
    cluname = clu[0]
    clunum = int(cluname[21:])-2  # index for retrieving fsa
    pathname = 'Z:\Dinghao\MiceExp\ANMD'+cluname[1:5]+'\\'+cluname[:14]+'\\'+cluname[:17]
    curr_spike_all = clu[1][0]
    
    # load bad trial indices
    beh_par_file = sio.loadmat(pathname+pathname[42:]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    curr_spike_all = clu[1][0]
    curr_spike_good = curr_spike_all[ind_good_beh]
    run_onset_mean = np.mean([trial[1250:6250]*1250 for trial in curr_spike_good], 
                             axis=0)
    run_onset_sem = sem([trial[1250:6250]*1250 for trial in curr_spike_good],
                        axis=0)
    
    # filename = input('file route: ')
    filestem = pathname+pathname[-18:]
    
    # import beh file
    run_bout_file_name = r'Z:\Dinghao\code_dinghao\run_bouts\fsa_run_bouts'+pathname[-18:]+'_BefRunBout0.mat'
    run_bout_file = mat73.loadmat(run_bout_file_name)
    
    times = run_bout_file['timeStepRun']
    fsa = run_bout_file['filteredSpikeArrayRunBoutOnSet'][clunum]  # bout x time
        
    peak_det = np.mean([trial[:6250] for trial in curr_spike_all], axis=0)
    if fsa.shape[0]==9201 or cluname in burst or cluname in burstv or cluname in other:
        pass
    else:
        fsa_mean = np.mean(fsa[:, 400:2000], axis=0)  # 2s around bout-onset
        fsa_sem = sem(fsa[:, 400:2000], axis=0)
        all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
        avg_ro.append(run_onset_mean)
        avg_rb.append(fsa_mean)
        burst_ro.append(np.mean(run_onset_mean[1750:2500]))  # .4s before onset
        burst_rb.append(np.mean(fsa_mean[560:800]))

sem_ro = sem(avg_ro, axis=0)
avg_ro = np.mean(avg_ro, axis=0)
sem_rb = sem(avg_rb, axis=0)
avg_rb = np.mean(avg_rb, axis=0)


#%% plotting all
tot_plots = len(all_ro_rb)
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()

rb_xaxis = np.arange(-800, 800)*.0025
ro_xaxis = np.arange(-2500, 2500)/1250

i = 0
for clu in list(all_ro_rb.items()):
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
        
    cluname = clu[0]
    ro_curr = all_ro_rb[cluname][0]
    ro_sem_curr = all_ro_rb[cluname][1]
    rb_curr = all_ro_rb[cluname][2]
    rb_sem_curr = all_ro_rb[cluname][3]
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
           xlim=(-2,2),
           title=cluname, 
           xlabel='time (s)',
           ylabel='spike rate (Hz)')
    ax.vlines(0, 0, 20, color='grey', alpha=.1)
    ax.legend([ro_ln, rb_ln], ['run onset', 'run bout'])
    
    i+=1

plt.subplots_adjust(hspace = 0.5)
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ro_rb_trough.png')


#%% plotting average
fig, ax = plt.subplots()

avg_ro_ln, = ax.plot(ro_xaxis, avg_ro, color='royalblue')
avg_rb_ln, = ax.plot(rb_xaxis, avg_rb, color='grey')
ax.fill_between(ro_xaxis,
                avg_ro+sem_ro,
                avg_ro-sem_ro,
                color='cornflowerblue',
                alpha=.1)
ax.fill_between(rb_xaxis,
                avg_rb+sem_rb,
                avg_rb-sem_rb,
                color='gainsboro',
                alpha=.1)
ax.set(xlim=(-2,2),
       ylim=(0,6),
       title='avg run-onset vs run-bout',
       xlabel='time (s)',
       ylabel='spike rate (Hz)')
ax.legend([avg_ro_ln, avg_rb_ln], ['run-onset', 'run-bout'])

plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ro_rb_trough_avg.png')


#%% t-test and plotting bar graph
t_res = ttest_rel(a=burst_ro, b=burst_rb)
pval = t_res[1]
print('t_test: {}'.format(t_res))

wilc = wilcoxon(burst_ro, burst_rb, alternative='two-sided')
pvalwilc = wilc[1]
print('Wilcoxon: {}'.format(wilc))

fig, ax = plt.subplots(figsize=(5,4))

x = [0, 3]; y = [0, 3]
ax.plot(x, y, color='grey')
ax.scatter(burst_ro, burst_rb, s=5, color='grey', alpha=.5)
mean_ro = np.mean(burst_ro); mean_rb = np.mean(burst_rb)
sem_ro = sem(burst_ro); sem_rb = sem(burst_ro)
ax.scatter(mean_ro, mean_rb, s=15, color='royalblue', alpha=.9)
ax.plot([mean_ro, mean_ro], 
        [mean_rb+sem_rb, mean_rb-sem_rb], 
        color='cornflowerblue', alpha=.7)
ax.plot([mean_ro+sem_ro, mean_ro-sem_ro], 
        [mean_rb, mean_rb], 
        color='cornflowerblue', alpha=.7)

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc),
       xlabel='run-onset',
       ylabel='run-bout-onset',
       xlim=(0.7,2.6), ylim=(0.7,2.6))

plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ro_rb_trough_avg_bivariate.png')


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
       ylabel='spike rate (Hz)',
       ylim=(0,2.8))

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_ro_rb_trough_avg_bar.png')