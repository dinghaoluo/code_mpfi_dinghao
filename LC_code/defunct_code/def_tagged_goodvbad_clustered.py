# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:20:55 2023

LC: visual and statistical comparison between good and bad trial bursts 

*use clustered data*

bad trial parameters 30 Jan 2023 (in getBehParameters()):
    rewarded == -1
    noStop
    noFullStop

@author: Dinghao Luo
"""

#%% imports
import os
import sys
import numpy as np
from scipy.stats import sem, ttest_rel, wilcoxon
import matplotlib.pyplot as plt 
import scipy.io as sio

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list, single_unit
pathLC = rec_list.pathLC

if ('Z:\Dinghao\code_dinghao\LC_tagged_by_sess' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\LC_tagged_by_sess')


#%% MAIN
all_tagged_train = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_all_tagged_info.npy',
                           allow_pickle=True).item()
clstr = np.load('Z:/Dinghao/code_dinghao/LC_all_tagged/LC_tagged_clustered_fromall.npy',
                allow_pickle=True).item()
burst = clstr['cluster 1']
burst_good = []; burst_bad = []
b_good = {}; b_good_sem = {}  # single trials
b_bad = {}; b_bad_sem = {}  # single trials
max_length = 13750  # max length for trial analysis

for pathname in pathLC:
    sessname = pathname[-17:]
    
    # import bad beh trial numbers
    beh_par_file = sio.loadmat(pathname+pathname[-18:]+
                               '_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat')
                                   # -1 to account for MATLAB Python difference
    ind_bad_beh = np.where(beh_par_file['behPar'][0]['indTrBadBeh'][0]==1)[1]-1
                                     # -1 to account for 0 being an empty trial
    ind_good_beh = np.arange(beh_par_file['behPar'][0]['indTrBadBeh'][0].shape[1]-1)
    ind_good_beh = np.delete(ind_good_beh, ind_bad_beh)
    
    # import tagged cell spike trains from all_tagged_train
    if len(ind_bad_beh) >= 15:  # 15 bad trials at least, prevents contam.
        for name in burst:
            if name[:17] == sessname:
                curr_tagged = all_tagged_train[name][0,:]  # train of current clu
                curr_good = np.zeros([len(ind_good_beh), max_length])
                curr_bad = np.zeros([len(ind_bad_beh), max_length])
                for i in range(len(ind_good_beh)):
                    curr_length = len(curr_tagged[ind_good_beh[i]])
                    curr_good[i, :curr_length] = curr_tagged[ind_good_beh[i]][:max_length]
                for i in range(len(ind_bad_beh)):
                    curr_length = len(curr_tagged[ind_bad_beh[i]])
                    curr_bad[i, :curr_length] = curr_tagged[ind_bad_beh[i]][:max_length]
                b_good[name] = np.mean(curr_good, axis=0)
                b_good_sem[name] = sem(curr_good, axis=0)
                burst_good.append(np.mean(b_good[name][3125:4375]))
                b_bad[name] = np.mean(curr_bad, axis=0)
                b_bad_sem[name] = sem(curr_bad, axis=0)
                burst_bad.append(np.mean(b_bad[name][3125:4375]))

b_g_avg = []
b_b_avg = []
for clu in list(b_good.items()):
    b_g_avg.append(clu[1])
for clu in list(b_bad.items()):
    b_b_avg.append(clu[1])
b_g_sem = sem(b_g_avg, axis=0)
b_g_avg = np.mean(b_g_avg, axis=0)
b_b_sem = sem(b_b_avg, axis=0)
b_b_avg = np.mean(b_b_avg, axis=0)


#%% plotting
print('\nplotting avg onset-bursting good vs bad spike trains...')
tot_plots = len(b_good)  # total number of cells
col_plots = 5
row_plots = tot_plots // col_plots
if tot_plots % col_plots != 0:
    row_plots += 1
plot_pos = np.arange(1, tot_plots+1)

fig = plt.figure(1, figsize=[5*4, row_plots*2.5]); fig.tight_layout()
xaxis = np.arange(-3750, 10000, 1)/1250 

for i in range(tot_plots):
    curr_clu_good = list(b_good.items())[i]
    curr_clu_name = curr_clu_good[0]
    curr_good_avg = curr_clu_good[1]
    curr_good_sem = b_good_sem[curr_clu_name]
    curr_bad_avg = b_bad[curr_clu_name]
    curr_bad_sem = b_bad[curr_clu_name]
    
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
    peak = single_unit.neu_peak_detection(curr_good_avg)
    if peak==True:
        ax.set_title(curr_clu_name[-22:], color='r', fontsize = 10)
    else:
        ax.set_title(curr_clu_name[-22:], fontsize = 10)
    ax.set(ylim=(0, np.max(curr_good_avg)*1250*1.5),
           ylabel='spike rate (Hz)',
           xlabel='time (s)')
    good_avg = ax.plot(xaxis, curr_good_avg*1250, color='seagreen')
    good_sem = ax.fill_between(xaxis, curr_good_avg*1250+curr_good_sem*1250,
                                      curr_good_avg*1250-curr_good_sem*1250,
                                      color='springgreen')
    bad_avg = ax.plot(xaxis, curr_bad_avg*1250, color='firebrick', alpha=.3)
    # bad_sem = ax.fill_between(xaxis, curr_bad_avg+curr_bad_sem,
    #                                  curr_bad_avg-curr_bad_sem,
    #                                  color='lightcoral')
    ax.vlines(0, 0, 20, color='grey', alpha=.25)

plt.subplots_adjust(hspace = 0.5)
plt.show()

out_directory = r'Z:\Dinghao\code_dinghao\LC_all_tagged'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)
fig.savefig(out_directory + '\\'+'LC_tagged_goodvbad_(alignedRun)_clstr1.png')


#%% avg profile for onset-bursting clus
print('\nplotting avg onset-bursting good vs bad averaged spike trains...')

b_g_burst_avg = []
b_b_burst_avg = []
for clu in list(b_good.items()):
    peak = single_unit.neu_peak_detection(clu[1])
    if peak==True:
        b_g_burst_avg.append(clu[1])
        b_b_burst_avg.append(b_bad[clu[0]])
b_g_burst_sem = sem(b_g_burst_avg, axis=0)
b_g_burst_avg = np.mean(b_g_burst_avg, axis=0)
b_b_burst_sem = sem(b_b_burst_avg, axis=0)
b_b_burst_avg = np.mean(b_b_burst_avg, axis=0)

fig, ax = plt.subplots(figsize=(5,4))
b_good_ln, = ax.plot(xaxis, b_g_burst_avg*1250, color='forestgreen')
b_bad_ln, = ax.plot(xaxis, b_b_burst_avg*1250, color='grey')
ax.fill_between(xaxis, b_g_burst_avg*1250+b_g_burst_sem*1250, 
                       b_g_burst_avg*1250-b_g_burst_sem*1250,
                       color='mediumseagreen', alpha=.1)
ax.fill_between(xaxis, b_b_burst_avg*1250+b_b_burst_sem*1250, 
                       b_b_burst_avg*1250-b_b_burst_sem*1250,
                       color='gainsboro', alpha=.1)
ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(title='good v bad trials (tagged cluster 1)',
       ylim=(2, 9),
       xlim=(-1,4),
       ylabel='spike rate (Hz)',
       xlabel='time (s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend([b_good_ln, b_bad_ln], 
          ['good trial', 'bad trial'])

fig.savefig(out_directory + '\\'+'LC_tagged_goodvbad_(alignedRun)_avg_clstr1.png',
            dpi=300,
            bbox_inches='tight')


#%% tests and bar graph
t_res = ttest_rel(a=burst_good, b=burst_bad)
pval = t_res[1]
print('t_test: {}'.format(t_res))

wilc = wilcoxon(burst_good, burst_bad, alternative='two-sided')
pvalwilc = wilc[1]
print('Wilcoxon: {}'.format(wilc))

bg_disp = [i*1250 for i in burst_good]
bb_disp = [i*1250 for i in burst_bad]

fig, ax = plt.subplots(figsize=(4,4))
# for p in ['top', 'right']:
#     ax.spines[p].set_visible(False)

x = [0, 11]; y = [0, 11]
ax.plot(x, y, color='grey')
ax.scatter(bg_disp, bb_disp, s=5, color='grey', alpha=.5)
mean_bg = np.mean(bg_disp); mean_bb = np.mean(bb_disp)
sem_bg = sem(bg_disp); sem_bb = sem(bb_disp)
ax.scatter(mean_bg, mean_bb, s=15, color='forestgreen', alpha=.9)
ax.plot([mean_bg, mean_bg], 
        [mean_bb+sem_bb, mean_bb-sem_bb], 
        color='mediumseagreen', alpha=.7)
ax.plot([mean_bg+sem_bg, mean_bg-sem_bg], 
        [mean_bb, mean_bb], 
        color='mediumseagreen', alpha=.7)

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc),
       xlabel='good trials',
       ylabel='bad trials',
       xlim=(1,10), ylim=(1,10))

plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_goodvbad_avg_clstr1_bivariate.png')


fig, ax = plt.subplots(figsize=(3,6))

xaxis = [1, 2]

ax.bar(xaxis, 
       [np.mean(bg_disp), np.mean(bb_disp)],
       yerr=[sem(bg_disp), sem(bb_disp)], capsize=5,
       width=0.8,
       tick_label=['good trials', 'bad trials'],
       edgecolor=['darkgreen', 'darkred'],
       color=(0,0,0,0))

ax.scatter(1+np.random.random(len(burst_good))*0.5-0.25, bg_disp,
           s=3, color='royalblue', alpha=.5)
ax.scatter(2+np.random.random(len(burst_bad))*0.5-0.25, bb_disp,
           s=3, color='grey', alpha=.5)

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc))

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_tagged_goodvbad_avg_clstr1_bar.png')


#%% histogram to couple with bivariate
fig, ax = plt.subplots(figsize=(6, 1))
ax.set(xlim=(-2, 2))
for p in ['top', 'left', 'right']:
    ax.spines[p].set_visible(False)
ax.set_yticks([])

pt_dist = []
for i in range(len(bg_disp)):
    x = bg_disp[i]; y = bb_disp[i]
    if x>y:
        pt_dist.append(np.sqrt((x-y)**2/2))
    elif x<y:
        pt_dist.append(-np.sqrt((x-y)**2/2))

bins = np.arange(-1, 1, .1)
ax.hist(pt_dist, bins=bins,
        color='forestgreen',
        edgecolor='grey')