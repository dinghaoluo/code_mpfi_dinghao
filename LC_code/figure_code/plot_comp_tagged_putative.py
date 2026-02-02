# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:25:56 2023
Modified on 28 Feb 

plot waveform and spike rate comparison between tagged and putative 

@author: Dinghao Luo
"""


#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from scipy.stats import mannwhitneyu

from common import mpl_formatting, colour_putative, colour_tagged, colour_other
mpl_formatting()


#%% paths
LC_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys')


#%% function
def _accumulate_info(df: pd.DataFrame,
                     identity: str,
                     method: str = 'FWHM') -> None:
    """
    compute and store spike rate, waveform asymmetry, and spike width from cluster dataframe.
    
    parameters:
    - df: pd.DataFrame
        dataframe containing spike cluster info; must have 'spike_rate' and 'waveform' columns
    - identity: str
        string key used to index into the global dictionaries for storing results
    - method: str (default: 'FWHM')
        method used to compute spike width; choose from:
        - 'FWHM': full width at half minimum (default, interpolated for robustness)
        - 'half_width': time between positive peaks before and after trough
    
    returns:
    - None
        results are appended to global dictionaries: spike_width_dict, asymmetry_dict, spike_rate_dict
    """
    if df.empty:
        raise Exception('DataFrame is empty; check file integrity.')
    
    for clu in df.itertuples():        
        spike_rate = clu.spike_rate
        waveform = clu.waveform 
        if np.isnan(waveform[0]):
            continue 

        # locate trough
        wf_min = np.argmin(waveform)

        # calculate asymmetry
        des = waveform[:wf_min]
        asc = waveform[wf_min:]
        des_max = np.argmax(des)
        asc_max = np.argmax(asc)
        a = des[des_max]
        b = asc[asc_max]
        asymmetry = (b - a) / (b + a)

        # choose spike width calculation method
        if method == 'half_width':
            spike_width = (asc_max + wf_min - des_max) / 20  # ms
        elif method == 'FWHM':
            # interpolate waveform
            x = np.arange(len(waveform))
            interp = interp1d(x, waveform, kind='cubic')
            x_hi = np.linspace(0, len(waveform) - 1, 320)  # 10x upsample
            wf_hi = interp(x_hi)
            
            trough_idx = np.argmin(wf_hi)
            trough_val = wf_hi[trough_idx]
            baseline = np.median(wf_hi[:10])  # adjust as needed
            half_max = (baseline + trough_val) / 2
            
            # find where waveform drops below half-max and then recovers
            left = np.where(wf_hi[:trough_idx] > half_max)[0]
            right = np.where(wf_hi[trough_idx:] > half_max)[0]
            
            if len(left) > 0 and len(right) > 0:
                spike_width = (right[0] + trough_idx - left[-1]) / 200  # ms
            else:
                spike_width = np.nan
        else:
            raise ValueError(f'unknown method: {method}')

        # plot and save figure after calculating spike width
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(waveform, label='waveform')
        ax.scatter(wf_min, waveform[wf_min], label='trough', color='black', zorder=5)

        if method == 'half_width':
            ax.scatter(des_max, waveform[des_max], label='pre-trough peak', color='blue', zorder=5)
            ax.scatter(asc_max + wf_min, waveform[asc_max + wf_min], label='post-trough peak', color='red', zorder=5)
        elif method == 'FWHM' and not np.isnan(spike_width):
            # show FWHM markers on upsampled waveform if you want; here we mark just the trough
            ax.axhline(half_max, linestyle='--', color='grey', label='half-max', zorder=1)

        ax.set(title=clu.Index)
        ax.legend(loc='best', fontsize='x-small')
        
        save_dir = LC_stem / 'single_cell_waveform' / 'spike_width_calculation_{method}'
        save_dir.mkdir(exist_ok=True)
        fig.savefig(
            save_dir / f'{clu.Index}.png',
            bbox_inches='tight')
        plt.close(fig)

        # append to output dicts
        asymmetry_dict[f'{identity}'].append(asymmetry)
        spike_width_dict[f'{identity}'].append(spike_width)
        spike_rate_dict[f'{identity}'].append(spike_rate)


#%% load data 
cell_profiles_path = LC_stem / 'LC_all_cell_profiles.pkl'
cell_profiles = pd.read_pickle(cell_profiles_path)

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

_accumulate_info(df_other, 'other')
_accumulate_info(df_tagged, 'tagged')
_accumulate_info(df_putative, 'putative')


#%% plotting
fig, ax = plt.subplots(figsize=(3, 2.5))

tgd  = ax.scatter(spike_width_dict['tagged'],
                  spike_rate_dict['tagged'], c=colour_tagged, ec='k',
                  s=10, lw=.1, alpha=.8, zorder=10)
pt   = ax.scatter(spike_width_dict['putative'],
                  spike_rate_dict['putative'], c=colour_putative, ec='k', 
                  s=10, lw=.1, alpha=.8, zorder=9)
ntgd = ax.scatter(spike_width_dict['other'],
                  spike_rate_dict['other'],
                  s=10, lw=.1, c=colour_other, ec='k', alpha=.8)

ax.set(ylabel='Firing rate (Hz)', xlabel='Spike width (ms)')

ax.legend([tgd, pt, ntgd], ['Tagged', 'Putative\nDbh+', 'Putative\nDbh-'], 
          frameon=False, fontsize=6)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
    
for ext in ('.png', '.pdf'):
    fig.savefig(
        LC_stem / 'UMAP' / 'summary_and_comparison' / f'spike_width_v_spike_rate{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% statistics (spike width)
spike_width_tagged   = [sw for sw in spike_width_dict['tagged'] if not np.isnan(sw)]
spike_width_putative = [sw for sw in spike_width_dict['putative'] if not np.isnan(sw)]
spike_width_other    = [sw for sw in spike_width_dict['other'] if not np.isnan(sw)]


print('\n--- Spike width (ms): median [Q1, Q3] ---')

med = np.median(spike_width_tagged)
q1, q3 = np.percentile(spike_width_tagged, [25, 75])
print(f'tagged:   {med:.3f} [{q1:.3f}, {q3:.3f}]')

med = np.median(spike_width_putative)
q1, q3 = np.percentile(spike_width_putative, [25, 75])
print(f'putative: {med:.3f} [{q1:.3f}, {q3:.3f}]')

med = np.median(spike_width_other)
q1, q3 = np.percentile(spike_width_other, [25, 75])
print(f'other:    {med:.3f} [{q1:.3f}, {q3:.3f}]')


#%% plotting (spike width)
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.set(ylabel='Spike width (ms)',
       xlim=(0,3.2))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['Tagged', 
                    'Putative\n$\it{Dbh}$+', 
                    'Putative\n$\it{Dbh}$-'])

bp = ax.boxplot(
    [spike_width_tagged, spike_width_putative, spike_width_other],
    positions=[.5, 1.5, 2.5],
    patch_artist=True,
    notch='True'
    )

jitter_tagged_spike_width_x = np.random.uniform(
    -.1, .1, len(spike_width_dict['tagged'])
    )
ax.scatter([.9]*len(spike_width_dict['tagged']) + jitter_tagged_spike_width_x, 
           spike_width_dict['tagged'], 
           s=10, c=colour_tagged, ec='k', lw=.1, alpha=.8)

jitter_putative_spike_width_x = np.random.uniform(
    -.1, .1, len(spike_width_dict['putative'])
    )
ax.scatter([1.9]*len(spike_width_dict['putative']) + jitter_putative_spike_width_x, 
           spike_width_dict['putative'], 
           s=10, c=colour_putative, ec='k', lw=.1, alpha=.8)

jitter_other_spike_width_x = np.random.uniform(
    -.1, .1, len(spike_width_dict['other'])
    )
ax.scatter([2.9]*len(spike_width_dict['other']) + jitter_other_spike_width_x, 
           spike_width_dict['other'], 
           s=10, c=colour_other, ec='none', lw=.1, alpha=.8)

colors = [colour_tagged, colour_putative, colour_other]
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
    spike_width_tagged, 
    spike_width_putative, 
    alternative='two-sided')
mwu_tagged_other = mannwhitneyu(
    spike_width_tagged, 
    spike_width_other, 
    alternative='two-sided')
mwu_putative_other = mannwhitneyu(
    spike_width_putative, 
    spike_width_other, 
    alternative='two-sided')

ymax = ax.get_ylim()[1]

for i, data in enumerate([spike_width_tagged,
                          spike_width_putative,
                          spike_width_other]):
    med = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    ax.text(0.5 + i, ymax * 0.95,
            f'{med:.2f}\n[{q1:.2f}, {q3:.2f}]',
            ha='center', va='top', fontsize=6)

ax.set(
       title=f'MWu p={mwu_tagged_putative[1]:.2g},'
             f'{mwu_tagged_other[1]:.2g},'
             f'{mwu_putative_other[1]:.2g}'
       )

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(
        LC_stem / 'UMAP' / 'summary_and_comparison' / f'spike_width_boxplot{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
    
#%% statistics (firing rate)
spike_rate_tagged_logged = [np.log(t) for t in spike_rate_dict['tagged']]
spike_rate_putative_logged = [np.log(t) for t in spike_rate_dict['putative']]
spike_rate_other_logged = [np.log(t) for t in spike_rate_dict['other']]


print('\n--- log spike rate (Hz): median [Q1, Q3] ---')

med = np.median(spike_rate_tagged_logged)
q1, q3 = np.percentile(spike_rate_tagged_logged, [25, 75])
print(f'tagged:   {med:.3f} [{q1:.3f}, {q3:.3f}]')

med = np.median(spike_rate_putative_logged)
q1, q3 = np.percentile(spike_rate_putative_logged, [25, 75])
print(f'putative: {med:.3f} [{q1:.3f}, {q3:.3f}]')

med = np.median(spike_rate_other_logged)
q1, q3 = np.percentile(spike_rate_other_logged, [25, 75])
print(f'other:    {med:.3f} [{q1:.3f}, {q3:.3f}]')


#%% plotting (firing rate)
fig, ax = plt.subplots(figsize=(2.8, 3))
ax.set(ylabel='ln(Firing rate) (Hz)',
       xlim=(0,3.2))
for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.set_xticklabels(['Tagged', 
                    'Putative\n$\it{Dbh}$+', 
                    'Putative\n$\it{Dbh}$-'])

bp = ax.boxplot(
    [spike_rate_tagged_logged, 
     spike_rate_putative_logged, 
     spike_rate_other_logged],
    positions=[.5, 1.5, 2.5],
    patch_artist=True,
    notch='True')

jitter_tagged_spike_rate_x = np.random.uniform(
    -.1, .1, len(spike_rate_tagged_logged)
    )
ax.scatter([.9]*len(spike_rate_tagged_logged) + jitter_tagged_spike_rate_x, 
           spike_rate_tagged_logged, 
           s=10, c=colour_tagged, ec='k', lw=.1, alpha=.8)

jitter_putative_spike_rate_x = np.random.uniform(
    -.1, .1, len(spike_rate_putative_logged)
    )
ax.scatter([1.9]*len(spike_rate_putative_logged) + jitter_putative_spike_rate_x, 
           spike_rate_putative_logged, 
           s=10, c=colour_putative, ec='k', lw=.1, alpha=.8)

jitter_other_spike_rate_x = np.random.uniform(
    -.1, .1, len(spike_rate_other_logged)
    )
ax.scatter([2.9]*len(spike_rate_other_logged) + jitter_other_spike_rate_x, 
           spike_rate_other_logged, 
           s=10, c=colour_other, ec='none', lw=.1, alpha=.8)

colors = ['royalblue', (0.055, 0.082, 0.502), 'grey']
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

ymax = ax.get_ylim()[1]

for i, data in enumerate([spike_rate_tagged_logged,
                           spike_rate_putative_logged,
                           spike_rate_other_logged]):
    med = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    ax.text(0.5 + i, ymax * 0.95,
            f'{med:.2f}\n[{q1:.2f}, {q3:.2f}]',
            ha='center', va='top', fontsize=6)

ax.set(
       title=rf'MWu p={mwu_tagged_putative[1]:.2g},'
             rf'{mwu_tagged_other[1]:.2g}, '
             rf'{mwu_tagged_other[1]:.2g}'
       )

fig.tight_layout()

for ext in ('.png', '.pdf'):
    fig.savefig(
        LC_stem / 'UMAP' / 'summary_and_comparison' / f'spike_rate_logged_boxplot{ext}',
        dpi=300,
        bbox_inches='tight'
        )
