# -*- coding: utf-8 -*-
"""
Created on Fri 13 Dec 08:59:31 2024

plot profiles of run-onset ON/OFF cells
plot profiles in good trials versus bad trials 
plot profiles in ctrl trials versus stim trials, 26 Dec 2024

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
from scipy.stats import sem 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys


#%% load dataframe 
df = pd.read_pickle(r'Z:\Dinghao\code_dinghao\HPC_ephys\HPC_all_profiles.pkl')


#%% functions
sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from plotting_functions import plot_violin_with_scatter
from common import normalise, normalise_to_all, mpl_formatting 
mpl_formatting()

def compute_mean_and_sem(arr):
    '''take a 2D matrix'''
    '''returns mean and sem along axis=0'''
    return np.nanmean(arr, axis=0), sem(arr, axis=0)

def compute_response_index(arr, ratiotype='ON',
                           pre=(int(3750-1250*1.5), int(3750-1250*.5)),
                           post=(int(3750+1250*.5), int(3750+1250*1.5))):
    '''take a 1D array'''
    '''return a float (pre/post)'''
    if ratiotype=='ON':
        return np.mean(arr[post[0]:post[1]])/np.mean(arr[pre[0]:pre[1]])
    elif ratiotype=='OFF':
        return np.mean(arr[pre[0]:pre[1]])/np.mean(arr[post[0]:post[1]])


#%% parameters
xaxis = np.arange(-1, 4, 1/1250)


#%% sort dataframe by baseline pre-post ratios 
df_sorted = df.sort_values(by='pre_post')


#%% matrices 
pop_mat = df_sorted['prof_mean'].to_numpy()
pop_mat = np.asarray([normalise(cell[2500:2500+5*1250]) for cell in pop_mat])

ON = df_sorted[df_sorted['class']=='run-onset ON']
OFF = df_sorted[df_sorted['class']=='run-onset OFF']


#%% overall plot 
fig, ax = plt.subplots(figsize=(2,2))

ax.imshow(pop_mat, aspect='auto', cmap='Greys', interpolation='none',
          extent=(-1, 4, 0, pop_mat.shape[0]))
ax.set(title=f'{ON.shape[0]} ON, {OFF.shape[0]} OFF',
       xlabel='time from run-onset (s)',
       ylabel='cell #')

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_run_onset_Greys{}'.
        format(ext),
        dpi=300,
        bbox_inches='tight'
        )
    
fig, ax = plt.subplots(figsize=(2.4,1.9))

cw = ax.imshow(pop_mat, aspect='auto', cmap='coolwarm', interpolation='none',
          extent=(-1, 4, 0, pop_mat.shape[0]))
ax.set(title=f'{ON.shape[0]} ON, {OFF.shape[0]} OFF',
       xlabel='time from run-onset (s)',
       ylabel='cell #')

plt.colorbar(cw, shrink=.5)

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_run_onset_coolwarm{}'.
        format(ext),
        dpi=300,
        bbox_inches='tight'
        )
    

#%% extract good and bad mean profiles 
OFF_mat = OFF['prof_mean'].to_numpy()
OFF_mat = np.asarray([normalise(cell[2500:2500+5*1250]) for cell in OFF_mat])

ON_mat = ON['prof_mean'].to_numpy()
ON_mat = np.asarray([normalise(cell[2500:2500+5*1250]) for cell in ON_mat])


#%% plotting  (ON and OFF)
fig, axs = plt.subplots(1, 2, figsize=(5,2))
plt.subplots_adjust(wspace=.5)

axs[0].imshow(ON_mat, aspect='auto', cmap='Greys', interpolation='none',
              extent=(-1, 4, 0, ON_mat.shape[0]))
axs[1].imshow(OFF_mat, aspect='auto', cmap='Greys', interpolation='none',
              extent=(-1, 4, 0, OFF_mat.shape[0]))

axs[0].set(title='run-onset ON')
axs[1].set(title='run-onset OFF')
for ax in axs:
    ax.set(xlabel='time from run-onset (s)', 
           ylabel='cell #')
    
for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_run_onset_ON_OFF_Greys{}'.
        format(ext),
        dpi=300, 
        bbox_inches='tight'
        )

fig, axs = plt.subplots(1, 2, figsize=(5,2))
plt.subplots_adjust(wspace=.5)

axs[0].imshow(ON_mat, aspect='auto', cmap='coolwarm', interpolation='none',
              extent=(-1, 4, 0, ON_mat.shape[0]))
axs[1].imshow(OFF_mat, aspect='auto', cmap='coolwarm', interpolation='none',
              extent=(-1, 4, 0, OFF_mat.shape[0]))

axs[0].set(title='run-onset ON')
axs[1].set(title='run-onset OFF')
for ax in axs:
    ax.set(xlabel='time from run-onset (s)', 
           ylabel='cell #')
    
for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_run_onset_ON_OFF_coolwarm{}'.
        format(ext),
        dpi=300, 
        bbox_inches='tight'
        )
    

#%% good v bad
'''
we are normalising each cell's good and bad trial profiles by the pooled (good+
bad trials) profiles, with the assumption that each cell's output in the CA1 
network is obviously tuned according to its spike rates
'''
ON_good_index = [compute_response_index(i, 'ON') for i in ON['prof_good_mean']]
ON_bad_index = [compute_response_index(i, 'ON') for i in ON['prof_bad_mean']]
OFF_good_index = [compute_response_index(i, 'OFF') for i in OFF['prof_good_mean']]
OFF_bad_index = [compute_response_index(i, 'OFF') for i in OFF['prof_bad_mean']]

ON_empty_mask = [good.size!=0 and bad.size!=0 for good, bad 
                 in zip(ON['prof_good_mean'].to_numpy(), ON['prof_bad_mean'].to_numpy())]

ON_good_mat = [cell[2500:2500+1250*5] for cell, valid 
               in zip(ON['prof_good_mean'].to_numpy(), ON_empty_mask)
               if valid]
ON_bad_mat = [cell[2500:2500+1250*5] for cell, valid 
              in zip(ON['prof_bad_mean'].to_numpy(), ON_empty_mask)
              if valid]

for i in range(len(ON_good_mat)):
    temp_pool = np.concatenate((ON_good_mat[i], ON_bad_mat[i]))
    ON_good_mat[i] = normalise_to_all(ON_good_mat[i], temp_pool)
    ON_bad_mat[i] = normalise_to_all(ON_bad_mat[i], temp_pool)
    
ON_good_mean, ON_good_error = compute_mean_and_sem(ON_good_mat)
ON_bad_mean, ON_bad_error = compute_mean_and_sem(ON_bad_mat)

OFF_empty_mask = [good.size!=0 and bad.size!=0 for good, bad 
                  in zip(OFF['prof_good_mean'].to_numpy(), OFF['prof_bad_mean'].to_numpy())]

OFF_good_mat = [cell[2500:2500+1250*5] for cell, valid 
                in zip(OFF['prof_good_mean'].to_numpy(), OFF_empty_mask)
                if valid]
OFF_bad_mat = [cell[2500:2500+1250*5] for cell, valid 
               in zip(OFF['prof_bad_mean'].to_numpy(), OFF_empty_mask) 
               if valid]

for i in range(len(OFF_good_mat)):
    temp_pool = np.concatenate((OFF_good_mat[i], OFF_bad_mat[i]))
    OFF_good_mat[i] = normalise_to_all(OFF_good_mat[i], temp_pool)
    OFF_bad_mat[i] = normalise_to_all(OFF_bad_mat[i], temp_pool)

OFF_good_mean, OFF_good_error = compute_mean_and_sem(OFF_good_mat)
OFF_bad_mean, OFF_bad_error = compute_mean_and_sem(OFF_bad_mat)


#%% plot good v bad profiles 
fig, axs = plt.subplots(1, 2, figsize=(3.6,1.2))
plt.subplots_adjust(wspace=.5)

lg, = axs[0].plot(xaxis, ON_good_mean, c='firebrick', linewidth=1, zorder=10)
axs[0].fill_between(xaxis, ON_good_mean+ON_good_error,
                           ON_good_mean-ON_good_error,
                    color='firebrick', edgecolor='none', alpha=.35, zorder=10)

lb, = axs[0].plot(xaxis, ON_bad_mean, c='grey', linewidth=1)
axs[0].fill_between(xaxis, ON_bad_mean+ON_bad_error,
                           ON_bad_mean-ON_bad_error,
                    color='grey', edgecolor='none', alpha=.35)

axs[0].legend([lg, lb], ['good trials', 'bad trials'], frameon=False, fontsize=5)

lg, = axs[1].plot(xaxis, OFF_good_mean, c='purple', linewidth=1, zorder=10)
axs[1].fill_between(xaxis, OFF_good_mean+OFF_good_error,
                           OFF_good_mean-OFF_good_error,
                    color='purple', edgecolor='none', alpha=.35, zorder=10)

lb, = axs[1].plot(xaxis, OFF_bad_mean, c='grey', linewidth=1)
axs[1].fill_between(xaxis, OFF_bad_mean+OFF_bad_error,
                           OFF_bad_mean-OFF_bad_error,
                    color='grey', edgecolor='none', alpha=.35)

axs[1].legend([lg, lb], ['good trials', 'bad trials'], frameon=False, fontsize=5)

axs[0].set(ylim=(.2,.5))
axs[1].set(ylim=(.24,.39))
for ax in axs:
    ax.set(xlabel='time from run-onset (s)', xticks=(0,2,4),
           ylabel='norm. spike rate')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_ON_OFF_good_bad{}'.
        format(ext),
        dpi=300, 
        bbox_inches='tight'
        )
    
    
#%% filter and plot indices 
'''filtering is performed with a threshold of 3*std'''
outlier_mask = [1 if 
                not (0 < good < 10) or 
                not (0 < bad < 10) 
                else 0
                for good, bad in zip(ON_good_index, ON_bad_index)]
ON_good_index_filt, ON_bad_index_filt = zip(*[(good, bad) for i, (good, bad) 
                                            in enumerate(zip(ON_good_index, ON_bad_index)) 
                                            if outlier_mask[i]==0])
plot_violin_with_scatter(ON_good_index_filt, ON_bad_index_filt, 'firebrick', 'grey',
                         xticklabels=('good\ntrials', 'bad\ntrials'),
                         ylabel='response index',
                         showscatter=False, plot_statistics=True,
                         ylim=(0, 5),
                         figsize=(1.4, 2),
                         save=True, 
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_ON_good_bad_index_violinplot')

outlier_mask = [1 if 
                not (0 < good < 10) or 
                not (0 < bad < 10) 
                else 0
                for good, bad in zip(OFF_good_index, OFF_bad_index)]
OFF_good_index_filt, OFF_bad_index_filt = zip(*[(good, bad) for i, (good, bad) 
                                            in enumerate(zip(OFF_good_index, OFF_bad_index)) 
                                            if outlier_mask[i]==0])
plot_violin_with_scatter(OFF_good_index_filt, OFF_bad_index_filt, 'purple', 'grey',
                         xticklabels=('good\ntrials', 'bad\ntrials'),
                         ylabel='response index',
                         showscatter=False, plot_statistics=True,
                         ylim=(0, 5),
                         figsize=(1.4, 2),
                         save=True, 
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\all_OFF_good_bad_index_violinplot')


#%% extract seperate experiments 
df_HPCLC = df[(df['rectype']=='HPCLC') & (df['cell_identity']=='pyr')]
df_HPCLCterm = df[(df['rectype']=='HPCLCterm') & (df['cell_identity']=='pyr')]


#%% plot HPCLC
count_ctrl_ON = (df_HPCLC['pre_post_ctrl']<.8).sum()
count_stim_ON = (df_HPCLC['pre_post_stim']<.8).sum()

df_sorted_ctrl = df_HPCLC.sort_values(by='pre_post_ctrl')
df_sorted_stim = df_HPCLC.sort_values(by='pre_post_stim')

ctrl_profiles = df_sorted_ctrl['prof_ctrl_mean'].to_numpy()
ctrl_profiles = [normalise(cell[2500:2500+5*1250]) for cell in ctrl_profiles]

stim_profiles = df_sorted_stim['prof_stim_mean'].to_numpy()
stim_profiles = [normalise(cell[2500:2500+5*1250]) for cell in stim_profiles]

fig, axs = plt.subplots(1, 2, figsize=(4.4,2.5))
axs[0].imshow(ctrl_profiles, extent=(-1, 4, 0, len(ctrl_profiles)),
              aspect='auto', cmap='Greys', interpolation='none')
axs[0].set(title='ctrl.')
axs[1].imshow(stim_profiles, extent=(-1, 4, 0, len(ctrl_profiles)),
              aspect='auto', cmap='Greys', interpolation='none')
axs[1].set(title='stim.')

for i in [0,1]:
    axs[i].set(xlabel='time from run-onset (s)',
               ylabel='cell #')
    
fig.suptitle(f'ctrl. ON: {count_ctrl_ON}, stim. ON: {count_stim_ON}')

fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLC_run_onset_ctrl_stim{}'
        .format(ext),
        dpi=300)


#%% percentage of ON cells within sessions 
perc_ON_ctrl_per_session = df_HPCLC.groupby('recname')['pre_post_ctrl'].apply(
    lambda x: (x < 0.8).mean() * 100  # mean of booleans = percentage of True values
).to_numpy()
perc_ON_stim_per_session = df_HPCLC.groupby('recname')['pre_post_stim'].apply(
    lambda x: (x < 0.8).mean() * 100
).to_numpy()

plot_violin_with_scatter(perc_ON_ctrl_per_session, perc_ON_stim_per_session,
                         'grey', 'firebrick',
                         paired=True,
                         alpha=.25,
                         dpi=300,
                         xticklabels=('ctrl.', 'stim.'),
                         title='ON',
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLC_ON_perc_sess')

perc_OFF_ctrl_per_session = df_HPCLC.groupby('recname')['pre_post_ctrl'].apply(
    lambda x: (x > 1.25).mean() * 100  # mean of booleans = percentage of True values
).to_numpy()
perc_OFF_stim_per_session = df_HPCLC.groupby('recname')['pre_post_stim'].apply(
    lambda x: (x > 1.25).mean() * 100
).to_numpy()

plot_violin_with_scatter(perc_OFF_ctrl_per_session, perc_OFF_stim_per_session,
                         'grey', 'purple',
                         paired=True,
                         alpha=.25,
                         dpi=300,
                         xticklabels=('ctrl.', 'stim.'),
                         title='OFF',
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLC_OFF_perc_sess')


#%% plot HPCLCterm
count_ctrl_term_ON = (df_HPCLCterm['pre_post_ctrl']<.8).sum()
count_stim_term_ON = (df_HPCLCterm['pre_post_stim']<.8).sum()

df_sorted_ctrl_term = df_HPCLCterm.sort_values(by='pre_post_ctrl')
df_sorted_stim_term = df_HPCLCterm.sort_values(by='pre_post_stim')

ctrl_term_profiles = df_sorted_ctrl_term['prof_ctrl_mean'].to_numpy()
ctrl_term_profiles = [normalise(cell[2500:2500+5*1250]) for cell in ctrl_term_profiles]

stim_term_profiles = df_sorted_stim_term['prof_stim_mean'].to_numpy()
stim_term_profiles = [normalise(cell[2500:2500+5*1250]) for cell in stim_term_profiles]

fig, axs = plt.subplots(1, 2, figsize=(4.4,2.5))
axs[0].imshow(ctrl_term_profiles, extent=(-1, 4, 0, len(ctrl_term_profiles)),
              aspect='auto', cmap='Greys', interpolation='none')
axs[0].set(title='ctrl.')
axs[1].imshow(stim_term_profiles, extent=(-1, 4, 0, len(ctrl_term_profiles)),
              aspect='auto', cmap='Greys', interpolation='none')
axs[1].set(title='stim.')

for i in [0,1]:
    axs[i].set(xlabel='time from run-onset (s)',
               ylabel='cell #')
    
fig.suptitle(f'ctrl. ON: {count_ctrl_term_ON}, stim. ON: {count_stim_term_ON}')
    
fig.tight_layout()

for ext in ['.png', '.pdf']:
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLCterm_run_onset_ctrl_stim{}'
        .format(ext),
        dpi=300)


#%% percentage of ON cells within sessions 
perc_ON_ctrl_term_per_session = df_HPCLCterm.groupby('recname')['pre_post_ctrl'].apply(
    lambda x: (x < 0.8).mean() * 100  # mean of booleans = percentage of True values
).to_numpy()
perc_ON_stim_term_per_session = df_HPCLCterm.groupby('recname')['pre_post_stim'].apply(
    lambda x: (x < 0.8).mean() * 100
).to_numpy()

plot_violin_with_scatter(perc_ON_ctrl_term_per_session, perc_ON_stim_term_per_session,
                         'grey', 'firebrick',
                         paired=True,
                         alpha=.25,
                         dpi=300,
                         xticklabels=('ctrl.', 'stim.'),
                         title='ON',
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLCterm_ON_perc_sess')

perc_OFF_ctrl_term_per_session = df_HPCLCterm.groupby('recname')['pre_post_ctrl'].apply(
    lambda x: (x > 1.25).mean() * 100  # mean of booleans = percentage of True values
).to_numpy()
perc_OFF_stim_term_per_session = df_HPCLCterm.groupby('recname')['pre_post_stim'].apply(
    lambda x: (x > 1.25).mean() * 100
).to_numpy()

plot_violin_with_scatter(perc_OFF_ctrl_term_per_session, perc_OFF_stim_term_per_session,
                         'grey', 'purple',
                         paired=True,
                         alpha=.25,
                         dpi=300,
                         xticklabels=('ctrl.', 'stim.'),
                         title='OFF',
                         save=True,
                         savepath=r'Z:\Dinghao\code_dinghao\HPC_ephys\run_onset_response\HPCLCterm_OFF_perc_sess')