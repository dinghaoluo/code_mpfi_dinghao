# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:00:55 2024
Modified for use on term activation data 5 Aug 2024

Quantify significantly responding cells for HPCLCterm activation

modification:
    - 17 May 2024, modified to include modulation index

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import sys 
from scipy.stats import ttest_rel, wilcoxon, sem, ranksums
import scipy.stats as st
from decimal import Decimal

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
pathHPC = rec_list.pathHPCLCtermopt


#%% dataframe to contain all results 
df = pd.read_pickle('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.pkl') 
MI = np.array(list(df['MI']))
MIe = np.array(list(df['MI_early'])) 
MIl = np.array(list(df['MI_late']))
shuf_MI = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_shuf_MI_pyr_only.npy', allow_pickle=True)
shuf_MIe = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_shuf_MIe_pyr_only.npy', allow_pickle=True)
shuf_MIl = np.load('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_shuf_MIl_pyr_only.npy', allow_pickle=True)
    

#%% plot shuffled MI distribution 
def gen_jit(data):
    return np.random.uniform(-.05, .05, len(data))
MI_perc = np.percentile(shuf_MI, [5, 95])
MIe_perc = np.percentile(shuf_MIe, [5, 95])
MIl_perc = np.percentile(shuf_MIl, [5, 95])

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
       title='modulation index (phasic LC-term activation)')


#%% initialisation
tot_excited = []
tot_inhibited = []

tot_seed = []

tot_clu = []

down_thres = 1.25
rise_thres = .8


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
rise_rise = []; rr=0
rise_down = []; rd=0
rise_other = []; ro=0
down_rise = []; dr=0
down_down = []; dd=0
down_other = []; do=0
other_rise = []; ori=0
other_down = []; od=0
other_other = []; oo=0
n_rise=0; n_down=0; n_other=0

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
        
        if n_rise!=0:
            rise_rise.append(rr/n_rise)
            rise_down.append(rd/n_rise)
            rise_other.append(ro/n_rise)
        if n_down!=0:
            down_down.append(dd/n_down)
            down_rise.append(dr/n_down)
            down_other.append(do/n_down)
        if n_other!=0:
            other_other.append(oo/n_other)
            other_rise.append(ori/n_other)
            other_down.append(od/n_other)
        
        # reset counters 
        recname = row['recname']
        excited = 0
        inhibited = 0
        rr=0; rd=0; ro=0; dd=0; dr=0; do=0; oo=0; ori=0; od=0
        n_rise=0; n_down=0; n_other=0
        
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
        n_down+=1
        down_ctrl.append(row['ctrl_mean'])
        # down_stim.append(row['stim_mean'])
        down_ctrl_change.append(np.mean(row['ctrl_mean'][2500:5000])/np.mean(row['ctrl_mean'][:2500]))
        # down_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
        if row['stim_pre_post']>down_thres:
            dd+=1
        elif row['stim_pre_post']<rise_thres:
            dr+=1
        else:
            do+=1
    elif row['ctrl_pre_post']<rise_thres:
        n_rise+=1
        rise_ctrl.append(row['ctrl_mean'])
        # rise_stim.append(row['stim_mean'])
        rise_ctrl_change.append(np.mean(row['ctrl_mean'][2500:5000])/np.mean(row['ctrl_mean'][:2500]))
        # rise_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
        if row['stim_pre_post']<rise_thres:
            rr+=1
        elif row['stim_pre_post']>down_thres:
            rd+=1
        else:
            ro+=1
    else:
        n_other+=1
        other_ctrl.append(row['ctrl_mean'])
        other_stim.append(row['stim_mean'])
        other_ctrl_change.append(np.mean(row['ctrl_mean'][2500:5000])/np.mean(row['ctrl_mean'][:2500]))
        other_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
        if row['stim_pre_post']>down_thres:
            od+=1
        elif row['stim_pre_post']<rise_thres:
            ori+=1
        else:
            oo+=1
            
    if row['stim_pre_post']>down_thres:
        down_stim.append(row['stim_mean'])
        down_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
    elif row['stim_pre_post']<rise_thres:
        rise_stim.append(row['stim_mean'])
        rise_stim_change.append(np.mean(row['stim_mean'][2500:5000])/np.mean(row['stim_mean'][:2500]))
            
    clu+=1


#%% change statistics
rise_rise_mean = np.mean(rise_rise); rise_rise_sem = sem(rise_rise)
rise_down_mean = np.mean(rise_down); rise_down_sem = sem(rise_down)
rise_other_mean = np.mean(rise_other); rise_other_sem = sem(rise_other)
down_down_mean = np.mean(down_down); down_down_sem = sem(down_down)
down_rise_mean = np.mean(down_rise); down_rise_sem = sem(down_rise)
down_other_mean = np.mean(down_other); down_other_sem = sem(down_other)
other_other_mean = np.mean(other_other); other_other_sem = sem(other_other)
other_rise_mean = np.mean(other_rise); other_rise_sem = sem(other_rise)
other_down_mean = np.mean(other_down); other_down_sem = sem(other_down)

# conf int
rise_rise_cint = (st.t.interval(confidence=.95, df=len(rise_rise)-1, loc=rise_rise_mean, scale=rise_rise_sem)-rise_rise_mean)[1]
rise_down_cint = (st.t.interval(confidence=.95, df=len(rise_down)-1, loc=rise_down_mean, scale=rise_down_sem)-rise_down_mean)[1]
rise_other_cint = (st.t.interval(confidence=.95, df=len(rise_other)-1, loc=rise_other_mean, scale=rise_other_sem)-rise_other_mean)[1]
down_down_cint = (st.t.interval(confidence=.95, df=len(down_down)-1, loc=down_down_mean, scale=down_down_sem)-down_down_mean)[1]
down_rise_cint = (st.t.interval(confidence=.95, df=len(down_rise)-1, loc=down_rise_mean, scale=down_rise_sem)-down_rise_mean)[1]
down_other_cint = (st.t.interval(confidence=.95, df=len(down_other)-1, loc=down_other_mean, scale=down_other_sem)-down_other_mean)[1]
other_other_cint = (st.t.interval(confidence=.95, df=len(other_other)-1, loc=other_other_mean, scale=other_other_sem)-other_other_mean)[1]
other_rise_cint = (st.t.interval(confidence=.95, df=len(other_rise)-1, loc=other_rise_mean, scale=other_rise_sem)-other_rise_mean)[1]
other_down_cint = (st.t.interval(confidence=.95, df=len(other_down)-1, loc=other_down_mean, scale=other_down_sem)-other_down_mean)[1]

# statistics 
rrdd = ranksums(rise_rise, down_down)[1]
rroo = ranksums(rise_rise, other_other)[1]
ddoo = ranksums(down_down, other_other)[1]


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

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_all_responsive_divided_bar.png',
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
       ylabel='spike rate (Hz)', yticks=[3,6,9],
       title='LC stim., CA1 RO-act.')
plt.legend([rc, rs], ['ctrl.', 'stim.'], frameon=False, fontsize=7, loc='upper right')
fig.tight_layout()
plt.show()
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_rise.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_rise.pdf',
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
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_down.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_down.pdf',
            bbox_inches='tight')


#%% overlap plots 
fig, ax = plt.subplots(figsize=(4.7,2.3))

colours = [(.55,0,0,1),(.55,0,0,.3),(.55,0,0,.3),
           (.72,.53,.04,1),(.72,.53,.04,.3),(.72,.53,.04,.3),
           (.41,.41,.41,1),(.41,.41,.41,.3),(.41,.41,.41,.3)]
x = [1,2,3,4,5,6,7,8,9]
y = [rise_rise_mean, rise_down_mean, rise_other_mean,
     down_down_mean, down_rise_mean, down_other_mean,
     other_other_mean, other_rise_mean, other_down_mean]
y_data = [rise_rise, rise_down, rise_other,
          down_down, down_rise, down_other,
          other_other, other_rise, other_down]
errors = [rise_rise_sem, rise_down_sem, rise_other_sem,
          down_down_sem, down_rise_sem, down_other_sem,
          other_other_sem, other_rise_sem, other_down_sem]
cint = [rise_rise_cint, rise_down_cint, rise_other_cint,
        down_down_cint, down_rise_cint, down_other_cint,
        other_other_cint, other_rise_cint, other_down_cint]

# USING CONFIDENCE INTERVAL FOR ERRORBARS
ax.bar(x, y,
       width=.8,
       color=colours)
ax.errorbar([pos+.05 for pos in x], y,
            cint,
            fmt='none', linewidth=1.5, color='k')
for i in range(9):
    ax.scatter([i+.95]*len(y_data[i]), y_data[i], c='grey', ec='none', s=3, alpha=.55)

ax.plot([1.2, 3.8], [1.08, 1.08], c='k', lw=.5)
ax.text(2.5, 1.11, '***p={}'.format('%.2E' % Decimal(rrdd)), ha='center', va='bottom', color='k', fontsize=6)

ax.plot([1.2, 6.8], [1.2, 1.2], c='k', lw=.5)
ax.text(4, 1.23, '*p={}'.format(round(rroo, 5)), ha='center', va='bottom', color='k', fontsize=6)

ax.plot([4.2, 6.8], [1.05, 1.05], c='k', lw=.5)
ax.text(5.5, 1.08, '***p={}'.format('%.2E' % Decimal(ddoo)), ha='center', va='bottom', color='k', fontsize=6)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.set(xticks=[1,2,3,4,5,6,7,8,9],
       xticklabels=['rise-\nrise', 'rise-\ndown', 'rise-\nother',
                    'down-\ndown', 'down-\nrise', 'down-\nother',
                    'other-\nother', 'other-\nrise', 'other-\ndown'],
       yticks=[0,.5,1],
       ylabel='proportion',
       title='cell ident. change')

fig.tight_layout()
plt.show()

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_identity_change_bar.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_identity_change_bar.pdf',
            bbox_inches='tight')


#%% change plot 
pval_r = ranksums(rise_ctrl_change, rise_stim_change)[1]

fig, ax = plt.subplots(figsize=(1,2))

# ax.plot([1, 2], [rise_ctrl_change, rise_stim_change], color='grey', lw=.1)
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

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_rel_rate_change_rise.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_rel_rate_change_rise.pdf',
            bbox_inches='tight')


pval_d = ranksums(down_ctrl_change, down_stim_change)[1]

fig, ax = plt.subplots(figsize=(1,2))

# ax.plot([1, 2], [down_ctrl_change, down_stim_change], color='grey', lw=.1)
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

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_rel_rate_change_down.png',
            dpi=500, bbox_inches='tight')
fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_ctrl_stim_rel_rate_change_down.pdf',
            bbox_inches='tight')