# -*- coding: utf-8 -*-
"""
Created on Sun June 11 13:12:54 2023

LC: visual and statistical comparison between good and bad trial RO peaks 

*use putative Dbh RO peaking cells*

bad trial parameters 30 Jan 2023 (in getBehParameters()):
    rewarded == -1
    noStop
    noFullStop

@author: Dinghao Luo
"""

#%% imports
import sys
import numpy as np
import pandas as pd
from scipy.stats import sem, ttest_rel, wilcoxon
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Arial' 
import scipy.io as sio
import mat73

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathLC = rec_list.pathLC


#%% load data 
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()
cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
tagged_RO_peaking_keys = []
putative_RO_peaking_keys = []
for cell in cell_prop.index:
    pk = cell_prop['peakness'][cell]  # peakness
    tg = cell_prop['tagged'][cell]  # tagged
    pt = cell_prop['putative'][cell]  # putative
    
    if pk and tg:
        tagged_RO_peaking_keys.append(cell)
    if pt:
        putative_RO_peaking_keys.append(cell)


#%% MAIN
all_ro_rb = {}

avg_ro = []; avg_rb = []
pk_ro = []; pk_rb = []

for clu in list(all_train.items()):
    cluname = clu[0]
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
    run_onset_mean = np.mean([trial[3025:5000]*1250 for trial in curr_spike_good if len(trial)>=5000], 
                             axis=0)
    run_onset_sem = sem([trial[3025:5000]*1250 for trial in curr_spike_good if len(trial)>=5000],
                        axis=0)
    
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
        rbsbef = np.mean(speed_all[start-1250:start])
        rbsaft = np.mean(speed_all[start:start+1250])
        if rbsaft>=ro_speed_aft_range[0] and rbsaft<=ro_speed_aft_range[1]:
            matched_bouts.append(i)

    times = run_bout_file['timeStepRun']
    fsa = run_bout_file['filteredSpikeArrayRunBoutOnSet'][clunum]  # bout x time
    
    # peak_det = np.mean([trial[:6250] for trial in curr_spike_all], axis=0)
    # if fsa.shape[0]==9201 or fsa.shape[0]<=50 or neu_peak_detection(peak_det)==False:
    #     pass
    if fsa.shape[0]==9201 or cluname not in tagged_RO_peaking_keys or len(matched_bouts)<=10:
        pass
    else:
        fsa_mean = np.mean(fsa[matched_bouts, 1000:1600], axis=0)  # -.5 to 1 s around bout-onset
        fsa_sem = sem(fsa[matched_bouts, 1000:1600], axis=0)
        all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
        avg_ro.append(run_onset_mean)
        avg_rb.append(fsa_mean)
        pk_ro.append(np.mean(run_onset_mean[413:1037]))  # .5s around onset
        pk_rb.append(np.mean(fsa_mean[100:300]))


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

rb_xaxis = np.arange(-200, 400)*.0025
ro_xaxis = np.arange(-725, 1250)/1250

i = 0
for clu in list(all_ro_rb.items()):
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
        
    cluname = clu[0]
    ro_curr = all_ro_rb[cluname][0][:]  # -1~4s
    ro_sem_curr = all_ro_rb[cluname][1][:]  # -1~4s
    rb_curr = all_ro_rb[cluname][2][:]  #-1~4s
    rb_sem_curr = all_ro_rb[cluname][3][:]  #-1~4s
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
           xlim=(-.5,1),
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

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_rovrb_tagged_ROpeaking.png',
            dpi=300,
            bbox_inches='tight')


#%% plotting average
print('\nplotting average spiking profiles...')
fig, ax = plt.subplots(figsize=(3,3))

for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

avg_ro_plot = avg_ro
sem_ro_plot = sem_ro
avg_rb_plot = avg_rb
sem_rb_plot = sem_rb

rb_xaxis = np.arange(-200, 400)*.0025
ro_xaxis = np.arange(-725, 1250)/1250

avg_ro_ln, = ax.plot(ro_xaxis, avg_ro_plot, color='royalblue')
avg_rb_ln, = ax.plot(rb_xaxis, avg_rb_plot, color='grey')
ax.fill_between(ro_xaxis,
                avg_ro_plot+sem_ro_plot,
                avg_ro_plot-sem_ro_plot,
                color='royalblue',
                alpha=.1)
ax.fill_between(rb_xaxis,
                avg_rb_plot+sem_rb_plot,
                avg_rb_plot-sem_rb_plot,
                color='grey',
                alpha=.1)
ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(xlim=(-.5,1),
       ylim=(1.8,5.5),
       yticks=[2,3,4,5],
       xlabel='time (s)',
       ylabel='spike rate (Hz)')
ax.legend([avg_ro_ln, avg_rb_ln], ['trial run onset', 'run-bout onset'], frameon=False, loc='upper right', fontsize=8)

fig.suptitle('trial run-onset v run-bout onset, tagged $\it{Dbh}$+')

plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_rovrb_avg_tagged_ROpeaking.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)


#%% t-test and plotting bar graph
t_res = ttest_rel(a=pk_ro, b=pk_rb)
pval = t_res[1]
print('t_test: {}'.format(t_res))

wilc = wilcoxon(pk_ro, pk_rb, alternative='two-sided')
pvalwilc = wilc[1]
print('Wilcoxon: {}'.format(wilc))

fig, ax = plt.subplots(figsize=(4,4))

x = [0, 20]; y = [0, 20]
ax.plot(x, y, color='grey', linestyle='dashed', alpha=.25)
ax.scatter(pk_ro, pk_rb, s=5, color='grey', alpha=.9)
mean_pro = np.mean(pk_ro); mean_prb = np.mean(pk_rb)
sem_pro = sem(pk_ro); sem_prb = sem(pk_ro)
ax.scatter(mean_pro, mean_prb, s=15, color='royalblue', alpha=.9)
ax.plot([mean_pro, mean_pro], 
        [mean_prb+sem_prb, mean_prb-sem_prb], 
        color='royalblue', alpha=.7)
ax.plot([mean_pro+sem_pro, mean_pro-sem_pro], 
        [mean_prb, mean_prb],
        color='royalblue', alpha=.7)

xmin=min(min(pk_ro), min(pk_rb))-.25
ymin=xmin
xmax=max(max(pk_ro), max(pk_rb))+.25
ymax=xmax

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc),
       xlabel='trial run onset spike rate (Hz)',
       ylabel='run-bout onset spike rate (Hz)',
       xlim=(xmin,xmax), ylim=(ymin,ymax))

plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all_tagged\LC_all_rovrb_avg_tagged_ROpeaking_bivariate.png',
            dpi=500,
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


#%% MAIN - putative
all_ro_rb = {}

avg_ro = []; avg_rb = []
pk_ro = []; pk_rb = []

for clu in list(all_train.items()):
    cluname = clu[0]
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
    run_onset_mean = np.mean([trial[3025:5000]*1250 for trial in curr_spike_good if len(trial)>=5000], 
                             axis=0)
    run_onset_sem = sem([trial[3025:5000]*1250 for trial in curr_spike_good if len(trial)>=5000],
                        axis=0)
    
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
        rbsbef = np.mean(speed_all[start-1250:start])
        rbsaft = np.mean(speed_all[start:start+1250])
        if rbsaft>=ro_speed_aft_range[0] and rbsaft<=ro_speed_aft_range[1]:
            matched_bouts.append(i)

    times = run_bout_file['timeStepRun']
    fsa = run_bout_file['filteredSpikeArrayRunBoutOnSet'][clunum]  # bout x time
    
    # peak_det = np.mean([trial[:6250] for trial in curr_spike_all], axis=0)
    # if fsa.shape[0]==9201 or fsa.shape[0]<=50 or neu_peak_detection(peak_det)==False:
    #     pass
    if fsa.shape[0]==9201 or cluname not in putative_RO_peaking_keys or len(matched_bouts)<=10:
        pass
    else:
        fsa_mean = np.mean(fsa[matched_bouts, 1000:1600], axis=0)  # -.5 to 1 s around bout-onset
        fsa_sem = sem(fsa[matched_bouts, 1000:1600], axis=0)
        all_ro_rb[cluname] = [run_onset_mean, run_onset_sem, fsa_mean, fsa_sem]
        avg_ro.append(run_onset_mean)
        avg_rb.append(fsa_mean)
        pk_ro.append(np.mean(run_onset_mean[413:1037]))  # .5s around onset
        pk_rb.append(np.mean(fsa_mean[300:500]))


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

rb_xaxis = np.arange(-200, 400)*.0025
ro_xaxis = np.arange(-725, 1250)/1250

i = 0
for clu in list(all_ro_rb.items()):
    ax = fig.add_subplot(row_plots, col_plots, plot_pos[i])
        
    cluname = clu[0]
    ro_curr = all_ro_rb[cluname][0][:]  # -1~4s
    ro_sem_curr = all_ro_rb[cluname][1][:]  # -1~4s
    rb_curr = all_ro_rb[cluname][2][:]  #-1~4s
    rb_sem_curr = all_ro_rb[cluname][3][:]  #-1~4s
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

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_putative_ROpeaking.png',
            dpi=500,
            bbox_inches='tight')


#%% plotting average
print('\nplotting average spiking profiles...')
fig, ax = plt.subplots(figsize=(3,3))

for p in ['right', 'top']:
    ax.spines[p].set_visible(False)

avg_ro_plot = avg_ro
sem_ro_plot = sem_ro
avg_rb_plot = avg_rb
sem_rb_plot = sem_rb

rb_xaxis = np.arange(-200, 400)*.0025
ro_xaxis = np.arange(-725, 1250)/1250

avg_ro_ln, = ax.plot(ro_xaxis, avg_ro_plot, color='orange')
avg_rb_ln, = ax.plot(rb_xaxis, avg_rb_plot, color='grey')
ax.fill_between(ro_xaxis,
                avg_ro_plot+sem_ro_plot,
                avg_ro_plot-sem_ro_plot,
                color='orange',
                alpha=.1)
ax.fill_between(rb_xaxis,
                avg_rb_plot+sem_rb_plot,
                avg_rb_plot-sem_rb_plot,
                color='grey',
                alpha=.1)
ax.vlines(0, 0, 10, color='grey', linestyle='dashed', alpha=.1)
ax.set(xlim=(-.5,1),
       ylim=(1.7,3.9),
       yticks=[2,3],
       xlabel='time (s)',
       ylabel='spike rate (Hz)')
ax.legend([avg_ro_ln, avg_rb_ln], ['trial run onset', 'run-bout onset'], frameon=False, loc='upper right', fontsize=8)

fig.suptitle('trial run-onset v run-bout onset, putative $\it{Dbh}$+')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_putative_ROpeaking.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)


#%% t-test and plotting bar graph
t_res = ttest_rel(a=pk_ro, b=pk_rb)
pval = t_res[1]
print('t_test: {}'.format(t_res))

wilc = wilcoxon(pk_ro, pk_rb, alternative='two-sided')
pvalwilc = wilc[1]
print('Wilcoxon: {}'.format(wilc))

fig, ax = plt.subplots(figsize=(4,4))

x = [0, 20]; y = [0, 20]
ax.plot(x, y, color='grey', linestyle='dashed', alpha=.25)
ax.scatter(pk_ro, pk_rb, s=5, color='grey', alpha=.9)
mean_pro = np.mean(pk_ro); mean_prb = np.mean(pk_rb)
sem_pro = sem(pk_ro); sem_prb = sem(pk_ro)
ax.scatter(mean_pro, mean_prb, s=15, color='orange', alpha=.9)
ax.plot([mean_pro, mean_pro], 
        [mean_prb+sem_prb, mean_prb-sem_prb], 
        color='orange', alpha=.7)
ax.plot([mean_pro+sem_pro, mean_pro-sem_pro], 
        [mean_prb, mean_prb],
        color='orange', alpha=.7)

xmin=min(min(pk_ro), min(pk_rb))-.25
ymin=xmin
xmax=max(max(pk_ro), max(pk_rb))+.25
ymax=xmax

ax.set(title='tt {}\nwilc {}'.format(pval,pvalwilc),
       xlabel='trial run onset spike rate (Hz)',
       ylabel='run-bout onset spike rate (Hz)',
       xlim=(xmin,xmax), ylim=(ymin,ymax))

plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\LC_all\LC_all_rovrb_avg_putative_ROpeaking_bivariate.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)
