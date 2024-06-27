# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:55 2024

Quantify significantly responding cells for HPCLC activation

modification:
    - 17 May 2024, modified to include modulation index

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import sys 
from scipy.stats import ttest_rel, wilcoxon, sem
# from ast import literal_eval

if ('Z:\Dinghao\code_dinghao\common' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao\common')
from common import normalise, normalise_to_all

# plotting parameters 
xaxis = np.arange(-1250, 5000)/1250
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% dataframe to contain all results 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl') 
MI = np.array(list(df['MI']))
MIe = np.array(list(df['MI_early'])) 
MIl = np.array(list(df['MI_late']))
shuf_MI = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_shuf_MI_pyr_only.npy', allow_pickle=True)
shuf_MIe = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_shuf_MIe_pyr_only.npy', allow_pickle=True)
shuf_MIl = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_shuf_MIl_pyr_only.npy', allow_pickle=True)
    

#%% plot shuffled MI distribution 
def gen_jit(data):
    return np.random.uniform(-.05, .05, len(data))
MI_perc = np.percentile(shuf_MI, [10, 90])
MIe_perc = np.percentile(shuf_MIe, [10, 90])
MIl_perc = np.percentile(shuf_MIl, [10, 90])

fig, ax = plt.subplots(figsize=(5,3))
ax.scatter([1]*len(shuf_MI)+gen_jit(shuf_MI), shuf_MI, s=.5, c='grey', ec='none', alpha=.025)
ax.scatter([2]*len(MI)+gen_jit(MI), MI ,s=.5, c='k', ec='none', alpha=.25)
ax.scatter([3]*len(shuf_MIe)+gen_jit(shuf_MIe), shuf_MIe, s=.5, c='grey', ec='none', alpha=.025)
ax.scatter([4]*len(MIe)+gen_jit(MIe), MIe ,s=.5, c='k', ec='none', alpha=.25)
ax.scatter([5]*len(shuf_MIl)+gen_jit(shuf_MIl), shuf_MIl, s=.5, c='grey', ec='none', alpha=.025)
ax.scatter([6]*len(MIl)+gen_jit(MIl), MIl ,s=.5, c='k', ec='none', alpha=.25)

# violin plots
vps = ax.violinplot([shuf_MI, MI, shuf_MIe, MIe, shuf_MIl, MIl], showmeans=True, showextrema=False)
colors = ['grey', 'k']
for i in [0,1,2,3,4,5]:
    vps['bodies'][i].set_color(colors[np.mod(i, 2)])
    # vps['bodies'][i].set_edgecolor('none')
vps['cmeans'].set(color='darkred')

# 5-, 95-percentile lines 
ax.plot([.75, 2.25], [MI_perc[0], MI_perc[0]], lw=1, ls='dashed', c='k', alpha=.5)
ax.plot([.75, 2.25], [MI_perc[1], MI_perc[1]], lw=1, ls='dashed', c='k', alpha=.5)
ax.plot([2.75, 4.25], [MIe_perc[0], MIe_perc[0]], lw=1, ls='dashed', c='k', alpha=.5)
ax.plot([2.75, 4.25], [MIe_perc[1], MIe_perc[1]], lw=1, ls='dashed', c='k', alpha=.5)
ax.plot([4.75, 6.25], [MIl_perc[0], MIl_perc[0]], lw=1, ls='dashed', c='k', alpha=.5)
ax.plot([4.75, 6.25], [MIl_perc[1], MIl_perc[1]], lw=1, ls='dashed', c='k', alpha=.5)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xticks=[1,2,3,4,5,6], xticklabels=['shuf. MI', 'MI\n(0.5~1.5 s)', 'shuf. MI\nearly', 'MI early\n(0~1 s)', 'shuf. MI\nlate', 'MI late\n(1~2 s)'],
       ylabel='modulation index',
       title='modulation index (phasic LC activation)')


#%% initialisation
tot_excited = []
tot_inhibited = []

tot_seed = []

tot_clu = []

down_thres = 1.5
rise_thres = .67


#%% iterate over all 
recname = df['recname'][0]  # initialise recname as 1st recording 

excited = 0
inhibited = 0

seeds = []

# recording names 
recs = []

# modified to include MI's of both categories 
excited_MI = []
inhibited_MI = []

excited_MIe = []
inhibited_MIe = []
excited_MIl = []
inhibited_MIl = []

# rise-down cell analysis
rise_ctrl = []
rise_stim = []
down_ctrl = []
down_stim = []
other_ctrl = []
other_stim = []

rise_ctrl_change = []
rise_stim_change = []
down_ctrl_change = []
down_stim_change = []
other_ctrl_change = []
other_stim_change = []

# overlap analysis 
rise_rise = 0
rise_down = 0
rise_other = 0
down_rise = 0
down_down = 0
down_other = 0
other_rise = 0
other_down = 0
other_other = 0

clu = 0

for cluname, row in df.iterrows():
    
    # if new recording session, first save counters to tot_lists, and then reset all counters
    if row['recname'] != recname:
        # save data to lists
        tot_excited.append(excited)
        tot_inhibited.append(inhibited)
        
        tot_seed.append(seeds)
        
        tot_clu.append(clu)
        
        recs.append(recname)
        
        # reset counters 
        recname = row['recname']
        excited = 0
        inhibited = 0
        
        seeds = []
        
        clu = 0
    
    if row['MI']>MI_perc[1] or row['MI']<MI_perc[0]:
        mi = row['MI']
        mie = row['MI_early']
        mil = row['MI_late']
        if row['excited']:
            excited+=1
            excited_MI.append(mi)
            excited_MIe.append(mie)
            excited_MIl.append(mil)
        else:
            inhibited+=1
            inhibited_MI.append(-mi)
            inhibited_MIe.append(-mie)
            inhibited_MIl.append(-mil)
        if row['ctrl_pre_post']>down_thres:
            down_ctrl.append(row['ctrl_mean'])
            down_stim.append(row['stim_mean'])
            down_ctrl_change.append(np.mean(row['ctrl_mean'][2500:5000])/np.mean(row['ctrl_mean'][:2500]))
            down_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
            if row['stim_pre_post']>down_thres:
                down_down+=1
            elif row['stim_pre_post']<rise_thres:
                down_rise+=1
            else:
                down_other+=1
        elif row['ctrl_pre_post']<rise_thres:
            rise_ctrl.append(row['ctrl_mean'])
            rise_stim.append(row['stim_mean'])
            rise_ctrl_change.append(np.mean(row['ctrl_mean'][2500:5000])/np.mean(row['ctrl_mean'][:2500]))
            rise_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
            if row['stim_pre_post']<rise_thres:
                rise_rise+=1
            elif row['stim_pre_post']>down_thres:
                rise_down+=1
            else:
                rise_other+=1
        else:
            other_ctrl.append(row['ctrl_mean'])
            other_stim.append(row['stim_mean'])
            other_ctrl_change.append(np.mean(row['ctrl_mean'][2500:5000])/np.mean(row['ctrl_mean'][:2500]))
            other_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
            if row['stim_pre_post']>down_thres:
                other_down+=1
            elif row['stim_pre_post']<rise_thres:
                other_rise+=1
            else:
                other_other+=1
            
    clu+=1


#%% proportions
prop_excited = np.array([i/j for i, j in zip(tot_excited, tot_clu)])
prop_inhibited = np.array([i/j for i, j in zip(tot_inhibited, tot_clu)])

ttest_p = np.round(ttest_rel(prop_excited, prop_inhibited)[1], 5)
wilc_p = np.round(wilcoxon(prop_excited, prop_inhibited)[1], 5)

fig, ax = plt.subplots(figsize=(1.5,3))

mean_excited = np.mean(prop_excited)
mean_inhibited = np.mean(prop_inhibited)

# make points of 0 slightly above 0 for visualisation (does not affect the bars or statistics)
prop_excited[np.where(prop_excited==0.0)[0]] = 0.01
prop_inhibited[np.where(prop_inhibited==0.0)[0]] = 0.01

# jitters for visualisation 
jitter_exc = np.random.uniform(-.1, .1, len(prop_excited))
jitter_inh = np.random.uniform(-.1, .1, len(prop_inhibited))

ax.bar(1, mean_excited, 0.5, color='white', edgecolor='orange')
ax.bar(2, mean_inhibited, 0.5, color='white', edgecolor='forestgreen')
ax.scatter([1]*len(prop_excited), prop_excited, s=8, c='none', ec='grey')
ax.scatter([2]*len(prop_inhibited), prop_inhibited, s=8, c='none', ec='grey')

for i in range(len(prop_excited)):
    ax.plot([1, 2], [prop_excited[i], prop_inhibited[i]], color='grey', lw=1, alpha=.5)

ax.set(ylim=(0,.6), xlim=(.5, 2.5),
       xticks=[1,2], xticklabels=['excited','inhibited'],
       title='ttest_rel p={}\nwilc p={}'.format(ttest_p, wilc_p))

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_term_stim_all_responsive_divided_bar.png',
            dpi=500,
            bbox_inches='tight')

plt.close(fig)


#%% MI analysis 
fig, ax = plt.subplots(figsize=(1.5,5))

vp = ax.violinplot([inhibited_MI, excited_MI],
                   positions=[1, 1], showextrema=False, showmeans=True)
vp['bodies'][0].set_color('forestgreen')
vp['bodies'][0].set_edgecolor('darkgreen')
vp['bodies'][1].set_color('orange')
vp['bodies'][1].set_edgecolor('darkorange')
for i in [0, 1]:
    vp['bodies'][i].set_alpha(.2)

jit1 = np.random.uniform(-.04, .04, len(excited_MI))
jit2 = np.random.uniform(-.04, .04, len(inhibited_MI))
ax.scatter([1]*len(excited_MI)+jit1, excited_MI, s=3, c='orange', lw=.2, ec='darkorange')
ax.scatter([1]*len(inhibited_MI)+jit2, inhibited_MI, s=3, c='forestgreen', lw=.2, ec='darkgreen')

ax.plot([.5, 1.5], [1, 1], color='grey', linestyle='dashed', linewidth=1)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(.65, 1.35), xticks=[],
       title='modulation index\n(significant only)')

fig.tight_layout()
plt.show()
plt.close(fig)


#%% early MI analysis 
fig, ax = plt.subplots(figsize=(1.5,5))

vp = ax.violinplot([inhibited_MIe, excited_MIe],
                   positions=[1, 1], showextrema=False, showmeans=True)
vp['bodies'][0].set_color('forestgreen')
vp['bodies'][0].set_edgecolor('darkgreen')
vp['bodies'][1].set_color('orange')
vp['bodies'][1].set_edgecolor('darkorange')
for i in [0, 1]:
    vp['bodies'][i].set_alpha(.2)

jit1 = np.random.uniform(-.04, .04, len(excited_MIe))
jit2 = np.random.uniform(-.04, .04, len(inhibited_MIe))
ax.scatter([1]*len(excited_MIe)+jit1, excited_MIe, s=3, c='orange', lw=.2, ec='darkorange')
ax.scatter([1]*len(inhibited_MIe)+jit2, inhibited_MIe, s=3, c='forestgreen', lw=.2, ec='darkgreen')

ax.plot([.5, 1.5], [1, 1], color='grey', linestyle='dashed', linewidth=1)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(.65, 1.35), xticks=[],
       title='modulation index (early)\n(significant only)')

fig.tight_layout()
plt.show()
plt.close(fig)


#%% late MI analysis 
fig, ax = plt.subplots(figsize=(1.5,5))

vp = ax.violinplot([inhibited_MIl, excited_MIl],
                   positions=[1, 1], showextrema=False, showmeans=True)
vp['bodies'][0].set_color('forestgreen')
vp['bodies'][0].set_edgecolor('darkgreen')
vp['bodies'][1].set_color('orange')
vp['bodies'][1].set_edgecolor('darkorange')
for i in [0, 1]:
    vp['bodies'][i].set_alpha(.2)

jit1 = np.random.uniform(-.04, .04, len(excited_MIl))
jit2 = np.random.uniform(-.04, .04, len(inhibited_MIl))
ax.scatter([1]*len(excited_MIl)+jit1, excited_MIl, s=3, c='orange', lw=.2, ec='darkorange')
ax.scatter([1]*len(inhibited_MIl)+jit2, inhibited_MIl, s=3, c='forestgreen', lw=.2, ec='darkgreen')

ax.plot([.5, 1.5], [1, 1], color='grey', linestyle='dashed', linewidth=1)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(.65, 1.35), xticks=[],
       title='modulation index (late)\n(significant only)')

fig.tight_layout()
plt.show()
plt.close(fig)


#%% rise down cell analysis 
tot_rise = len(rise_ctrl); tot_down = len(down_ctrl); tot_other = len(other_ctrl)
# for clu in range(tot_rise):
#     full_data = np.concatenate((rise_ctrl[clu], rise_stim[clu]))
#     rise_ctrl[clu] = normalise_to_all(rise_ctrl[clu], full_data)
#     rise_stim[clu] = normalise_to_all(rise_stim[clu], full_data)
# for clu in range(tot_down):
#     full_data = np.concatenate((down_ctrl[clu], down_stim[clu]))
#     down_ctrl[clu] = normalise_to_all(down_ctrl[clu], full_data)
#     down_stim[clu] = normalise_to_all(down_stim[clu], full_data)

rise_ctrl_mean = np.mean(rise_ctrl, axis=0)
rise_ctrl_sem = sem(rise_ctrl, axis=0)
rise_stim_mean = np.mean(rise_stim, axis=0)
rise_stim_sem = sem(rise_stim, axis=0)
down_ctrl_mean = np.mean(down_ctrl, axis=0)
down_ctrl_sem = sem(down_ctrl, axis=0)
down_stim_mean = np.mean(down_stim, axis=0)
down_stim_sem = sem(down_stim, axis=0)


#%% rise down plot 
xaxis = (np.arange(6*1250)-2*1250)/1250

# rise cells 
fig, ax = plt.subplots(figsize=(2.3,2))
rc, = ax.plot(xaxis, rise_ctrl_mean, c='grey')
ax.fill_between(xaxis, rise_ctrl_mean+rise_ctrl_sem,
                       rise_ctrl_mean-rise_ctrl_sem,
                alpha=.2, color='grey', edgecolor='none')
rs, = ax.plot(xaxis, rise_stim_mean, c='royalblue')
ax.fill_between(xaxis, rise_stim_mean+rise_stim_sem,
                       rise_stim_mean-rise_stim_sem,
                alpha=.2, color='royalblue', edgecolor='none')
ax.set(xlabel='time (s)', xlim=(-1,4), xticks=[0,2,4], 
       ylabel='spike rate (Hz)', yticks=[2,4],
       title='LC stim., CA1 RO-act.')
plt.legend([rc, rs], ['ctrl.', 'stim.'], frameon=False, fontsize=7, loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_rise.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_rise.pdf',
            bbox_inches='tight')

# down cells 
fig, ax = plt.subplots(figsize=(2.3,2))
dc, = ax.plot(xaxis, down_ctrl_mean, c='grey')
ax.fill_between(xaxis, down_ctrl_mean+down_ctrl_sem,
                       down_ctrl_mean-down_ctrl_sem,
                alpha=.2, color='grey', edgecolor='none')
ds, = ax.plot(xaxis, down_stim_mean, c='royalblue')
ax.fill_between(xaxis, down_stim_mean+down_stim_sem,
                       down_stim_mean-down_stim_sem,
                alpha=.2, color='royalblue', edgecolor='none')
ax.set(xlabel='time (s)', xlim=(-1,4), xticks=[0,2,4], 
       ylabel='spike rate (Hz)', yticks=[1,2,3],
       title='LC stim., CA1 RO-inh.')
plt.legend([dc, ds], ['ctrl.', 'stim.'], frameon=False, fontsize=7, loc='lower right')
fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_down.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_down.pdf',
            bbox_inches='tight')


#%% overlap plots 
fig, ax = plt.subplots(figsize=(9,2))

ax.bar([1,2,3,4,5,6,7,8,9], [rise_rise/tot_rise, rise_down/tot_rise, rise_other/tot_rise,
                             down_down/tot_down, down_rise/tot_down, down_other/tot_down,
                             other_rise/tot_other, other_down/tot_other, other_other/tot_other])
ax.set(xticks=[1,2,3,4,5,6,7,8,9],
       xticklabels=['rise-rise', 'rise-down', 'rise-other',
                    'down-down', 'down-rise', 'down-other',
                    'other-rise', 'other-down', 'other-other'])

#%% change plot 
pval_r = ttest_rel(rise_ctrl_change, rise_stim_change)[1]

fig, ax = plt.subplots(figsize=(1,2))

ax.plot([1, 2], [rise_ctrl_change, rise_stim_change], color='grey', lw=.1)
vpr = ax.violinplot([rise_ctrl_change, rise_stim_change], showmeans=True, showextrema=False)
ax.set(xticks=[1,2], xticklabels=['ctrl.', 'stim.'],
       title='(0~2 s)/(-2~0 s)\nRO-act.',
       ylabel='fold',
       ylim=(-2, 42))

vpr['bodies'][0].set_color('grey')
vpr['bodies'][1].set_color('royalblue')
vpr['bodies'][0].set_edgecolor('k')
vpr['bodies'][1].set_edgecolor('k')
vpr['cmeans'].set_color('darkred')

plt.plot([1, 1, 2, 2], [38, 39, 39, 38], c='k', lw=.5)
plt.text(1.5, 39, 'p={}'.format(round(pval_r, 8)), ha='center', va='bottom', color='k', fontsize=5)

# fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_rel_rate_change_rise.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_rel_rate_change_rise.pdf',
            bbox_inches='tight')


pval_d = ttest_rel(down_ctrl_change, down_stim_change)[1]

fig, ax = plt.subplots(figsize=(1,2))

ax.plot([1, 2], [down_ctrl_change, down_stim_change], color='grey', lw=.1)
vpd = ax.violinplot([down_ctrl_change, down_stim_change], showmeans=True, showextrema=False)
ax.set(xticks=[1,2], xticklabels=['ctrl.', 'stim.'],
       title='(0~2 s)/(-2~0 s)\nRO-inh.',
       ylabel='fold',
       ylim=(0, 4.2))

vpd['bodies'][0].set_color('grey')
vpd['bodies'][1].set_color('royalblue')
vpd['bodies'][0].set_edgecolor('k')
vpd['bodies'][1].set_edgecolor('k')
vpd['cmeans'].set_color('darkred')

plt.plot([1, 1, 2, 2], [3.8, 3.9, 3.9, 3.8], c='k', lw=.5)
plt.text(1.5, 3.9, 'p={}'.format(round(pval_d, 8)), ha='center', va='bottom', color='k', fontsize=5)

# fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_rel_rate_change_down.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_ctrl_stim_rel_rate_change_down.pdf',
            bbox_inches='tight')
