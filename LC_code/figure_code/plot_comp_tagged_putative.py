# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:25:56 2023
Modified on 28 Feb 

plot waveform and spike rate comparison between tagged and putative 

@author: Dinghao Luo
"""


#%% imports
import numpy as np
import sys  
import matplotlib.pyplot as plt  
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.append(r'Z:\Dinghao\code_dinghao_mpfi\utils')
from common import mpl_formatting 
mpl_formatting()


#%% function
def accumulate_info(df: pd.DataFrame,
                    identity: str) -> None:
    if df.empty:
        Exception('DataFrame is empty; check file integrity.')
    
    for clu in df.itertuples():        
        # get spike rate 
        spike_rate = clu.spike_rate
        
        # get waveform for further calculation
        waveform = clu.waveform 
        if np.isnan(waveform[0]):
            continue 
    
        wf_min = np.argmin(waveform)  # find index of minimum point (trough)
        des = waveform[:wf_min]; asc = waveform[wf_min:]
        des_max = np.argmax(des); asc_max = np.argmax(asc)  # A and B
        a = des[des_max]; b = asc[asc_max]
        
        # finalise 
        spike_width = (asc_max+wf_min-des_max) * 50  # *50 to convert to μsec
        asymmetry = (b-a)/(b+a)
        
        # appending to lists 
        asymmetry_dict[f'{identity}'].append(asymmetry)
        spike_width_dict[f'{identity}'].append(spike_width)
        spike_rate_dict[f'{identity}'].append(spike_rate)


#%% load data 
cell_profiles = pd.read_pickle(
    'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )

df_tagged = cell_profiles[cell_profiles['identity']=='tagged']
df_putative = cell_profiles[cell_profiles['identity']=='putative']
df_other = cell_profiles[cell_profiles['identity']=='other']


#%% MAIN
# waveform asymmetry (B-A)/(B+A)
asymmetry_dict = {
    'other': [],
    'tagged': [],
    'putative': []
    }

# trough to AHP
spike_width_dict = {
    'other': [],
    'tagged': [],
    'putative': []
    }

# spike rate 
spike_rate_dict = {
    'other': [],
    'tagged': [],
    'putative': []
    }

accumulate_info(df_other, 'other')
accumulate_info(df_tagged, 'tagged')
accumulate_info(df_putative, 'putative')


#%% plotting
fig, ax = plt.subplots(figsize=(3, 2.5))

tgd = ax.scatter(spike_width_dict['tagged'],
                 spike_rate_dict['tagged'], c='royalblue', ec='k',
                 s=10, lw=.1, alpha=.8, zorder=10)
pt = ax.scatter(spike_width_dict['putative'],
                spike_rate_dict['putative'], c='orange', ec='k', 
                s=10, lw=.1, alpha=.8, zorder=9)
ntgd = ax.scatter(spike_width_dict['other'],
                  spike_rate_dict['other'],
                  s=10, lw=.1, c='grey', ec='k', alpha=.8)
ax.set(ylabel='spike rate (Hz)', xlabel='spike-width (μs)')
ax.legend([tgd, pt, ntgd], ['tagged', 'putative\nDbh+', 'putative\nDbh-'], 
          frameon=False, fontsize=6)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
    
for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP'
                rf'\summary_and_comparison\spike_width_v_spike_rate{ext}',
                dpi=300,
                bbox_inches='tight')

plt.close(fig)


#%% box plot for spike width
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.set(ylabel='spike-width (μs)',
       xlim=(0,3.2))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['tagged', 'putative\nDbh+', 'putative\nDbh-'])

bp = ax.boxplot(
    [spike_width_dict['tagged'], 
     spike_width_dict['putative'], 
     spike_width_dict['other']],
    positions=[.5, 1.5, 2.5],
    patch_artist=True,
    notch='True')

jitter_tagged_spike_width_x = np.random.uniform(
    -.1, .1, len(spike_width_dict['tagged'])
    )
ax.scatter([.9]*len(spike_width_dict['tagged']) + jitter_tagged_spike_width_x, 
           spike_width_dict['tagged'], 
           s=10, c='royalblue', ec='k', lw=.1, alpha=.8)

jitter_putative_spike_width_x = np.random.uniform(
    -.1, .1, len(spike_width_dict['putative'])
    )
ax.scatter([1.9]*len(spike_width_dict['putative']) + jitter_putative_spike_width_x, 
           spike_width_dict['putative'], 
           s=10, c='orange', ec='k', lw=.1, alpha=.8)

jitter_other_spike_width_x = np.random.uniform(
    -.1, .1, len(spike_width_dict['other'])
    )
ax.scatter([2.9]*len(spike_width_dict['other']) + jitter_other_spike_width_x, 
           spike_width_dict['other'], 
           s=10, c='grey', ec='none', lw=.1, alpha=.8)

colors = ['royalblue', 'orange', 'grey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][2].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)
    
mwu_tagged_putative = mannwhitneyu(
    spike_width_dict['tagged'], 
    spike_width_dict['putative'], 
    alternative='two-sided')
mwu_tagged_other = mannwhitneyu(
    spike_width_dict['tagged'], 
    spike_width_dict['other'], 
    alternative='two-sided')
mwu_putative_other = mannwhitneyu(
    spike_width_dict['putative'], 
    spike_width_dict['other'], 
    alternative='two-sided')

ax.set(
       title=rf'MWu p={round(mwu_tagged_putative[1], 5)},'
             rf'{round(mwu_tagged_other[1],5)}, '
             rf'{round(mwu_tagged_other[1],5)}'
       )

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP'
                rf'\summary_and_comparison\spike_width_boxplot{ext}',
                dpi=300,
                bbox_inches='tight')
    
plt.close(fig)

    
#%% box plot for spike rate 
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.set(ylabel='spike rate (Hz)',
       xlim=(0,3.2))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['tagged', 'putative\nDbh+', 'putative\nDbh-'])

bp = ax.boxplot(
    [spike_rate_dict['tagged'], 
     spike_rate_dict['putative'], 
     spike_rate_dict['other']],
    positions=[.5, 1.5, 2.5],
    patch_artist=True,
    notch='True')

jitter_tagged_spike_rate_x = np.random.uniform(
    -.1, .1, len(spike_rate_dict['tagged'])
    )
ax.scatter([.9]*len(spike_rate_dict['tagged']) + jitter_tagged_spike_rate_x, 
           spike_rate_dict['tagged'], 
           s=10, c='royalblue', ec='k', lw=.1, alpha=.8)

jitter_putative_spike_rate_x = np.random.uniform(
    -.1, .1, len(spike_rate_dict['putative'])
    )
ax.scatter([1.9]*len(spike_rate_dict['putative']) + jitter_putative_spike_rate_x, 
           spike_rate_dict['putative'], 
           s=10, c='orange', ec='k', lw=.1, alpha=.8)

jitter_other_spike_rate_x = np.random.uniform(
    -.1, .1, len(spike_rate_dict['other'])
    )
ax.scatter([2.9]*len(spike_rate_dict['other']) + jitter_other_spike_rate_x, 
           spike_rate_dict['other'], 
           s=10, c='grey', ec='none', lw=.1, alpha=.8)

colors = ['royalblue', 'orange', 'grey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

bp['fliers'][0].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][1].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)
bp['fliers'][2].set(marker ='o',
                color ='#e7298a',
                markersize=2,
                alpha=0.5)

for median in bp['medians']:
    median.set(color='darkred',
                linewidth=1)
    
mwu_tagged_putative = mannwhitneyu(
    spike_rate_dict['tagged'], 
    spike_rate_dict['putative'], 
    alternative='two-sided')
mwu_tagged_other = mannwhitneyu(
    spike_rate_dict['tagged'], 
    spike_rate_dict['other'], 
    alternative='two-sided')
mwu_putative_other = mannwhitneyu(
    spike_rate_dict['putative'], 
    spike_rate_dict['other'], 
    alternative='two-sided')

ax.set(
       title=rf'MWu p={round(mwu_tagged_putative[1], 5)},'
             rf'{round(mwu_tagged_other[1], 5)}, '
             rf'{round(mwu_tagged_other[1], 5)}'
       )

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\UMAP'
                rf'\summary_and_comparison\spike_rate_boxplot{ext}',
                dpi=300,
                bbox_inches='tight')
    
plt.close(fig)