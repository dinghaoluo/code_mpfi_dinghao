# -*- coding: utf-8 -*-
"""
Created on Fri 28 Feb 15:55:22 2025

plot ACGs as heatmaps

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import sys 

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting, normalise, gaussian_kernel_unity
mpl_formatting()

gaussian = gaussian_kernel_unity(sigma=2)

xaxis = np.arange(-200, 201)


#%% load data
cell_profiles = pd.read_pickle(
    r'Z:\Dinghao\code_dinghao\LC_ephys\LC_all_cell_profiles.pkl'
    )


#%% plot single-cell ACGs
for clu in cell_profiles.itertuples():
    cluname = clu.Index
    
    acg = clu.acg
    identity = clu.identity
    
    bins = np.arange(-200, 201)
    frequencies = np.convolve(acg, gaussian, mode='same')[9800:10200]
    
    bin_centres = (bins[:-1] + bins[1:]) / 2
    bin_width = np.diff(bins)
    
    fig, ax = plt.subplots(figsize=(2,1.6))
    ax.bar(bin_centres,
           frequencies,
           width=bin_width,
           align='center',
           color='k',
           alpha=.5)
    
    ax.set(title=cluname,
           xlabel='lag (ms)',
           ylabel='', yticks=())
    
    for s in ('left', 'top', 'right'):
        ax.spines[s].set_visible(False)
    
    for ext in ('.png', '.pdf'):
        fig.savefig(
            r'Z:\Dinghao\code_dinghao\LC_ephys'
            rf'\single_cell_ACGs\{cluname} {identity}{ext}',
            dpi=300,
            bbox_inches='tight'
            )
    
    plt.close(fig)
        

#%% population ACGs
df_tagged = cell_profiles[cell_profiles['identity']=='tagged'].copy()
df_tagged['acg_truncated'] = df_tagged['acg'].apply(
    lambda x: normalise(
        np.convolve(x, gaussian, mode='same')[9800:10201]
        )
    )
df_tagged['acg_trunc_for_argmax'] = df_tagged['acg'].apply(
    lambda x: x[9950:10001]
    )
df_tagged['argmax'] = df_tagged['acg_trunc_for_argmax'].apply(np.mean)
df_tagged_sorted = df_tagged.sort_values(by='argmax')
acgs_tagged = np.vstack(
    df_tagged_sorted['acg_truncated'].to_numpy()
    )

df_putative = cell_profiles[cell_profiles['identity']=='putative'].copy()
df_putative['acg_truncated'] = df_putative['acg'].apply(
    lambda x: normalise(
        np.convolve(x, gaussian, mode='same')[9800:10201]
        )
    )
df_putative['acg_trunc_for_argmax'] = df_putative['acg'].apply(
    lambda x: x[9950:10001]
    )
df_putative['argmax'] = df_putative['acg_trunc_for_argmax'].apply(np.mean)
df_putative_sorted = df_putative.sort_values(by='argmax')
acgs_putative = np.vstack(
    df_putative_sorted['acg_truncated'].to_numpy()
    )

df_other = cell_profiles[cell_profiles['identity']=='other'].copy()
df_other['acg_truncated'] = df_other['acg'].apply(
    lambda x: normalise(
        np.convolve(x, gaussian, mode='same')[9800:10201]
        )
    )
df_other['acg_trunc_for_argmax'] = df_other['acg'].apply(
    lambda x: x[9950:10001]
    )
df_other['argmax'] = df_other['acg_trunc_for_argmax'].apply(np.mean)
df_other_sorted = df_other.sort_values(by='argmax')
acgs_other = np.vstack(
    df_other_sorted['acg_truncated'].to_numpy()
    )


#%% mean traces
acgs_tagged_mean = np.mean(acgs_tagged, axis=0)
fig, ax = plt.subplots(figsize=(2,1.5))
ax.plot(xaxis, acgs_tagged_mean,
        color='k', linewidth=2)
ax.set(title='tagged Dbh+',
       xlabel='lag (ms)',
       ylabel='', yticks=())
for s in ('left', 'top', 'right'):
    ax.spines[s].set_visible(False)
for ext in ('.png', '.pdf'):
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LC_ephys'
        rf'\population_ACGs\tagged_mean{ext}',
        dpi=300,
        bbox_inches='tight'
        )
    
acgs_putative_mean = np.mean(acgs_putative, axis=0)
fig, ax = plt.subplots(figsize=(2,1.5))
ax.plot(xaxis, acgs_putative_mean,
        color='k', linewidth=2)
ax.set(title='putative Dbh+',
       xlabel='lag (ms)',
       ylabel='', yticks=())
for s in ('left', 'top', 'right'):
    ax.spines[s].set_visible(False)
for ext in ('.png', '.pdf'):
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LC_ephys'
        rf'\population_ACGs\putative_mean{ext}',
        dpi=300,
        bbox_inches='tight'
        )

acgs_other_mean = np.mean(acgs_other, axis=0)
fig, ax = plt.subplots(figsize=(2,1.5))
ax.plot(xaxis, acgs_other_mean,
        color='k', linewidth=2)
ax.set(title='putative Dbh-',
       xlabel='lag (ms)',
       ylabel='', yticks=())
for s in ('left', 'top', 'right'):
    ax.spines[s].set_visible(False)
for ext in ('.png', '.pdf'):
    fig.savefig(
        r'Z:\Dinghao\code_dinghao\LC_ephys'
        rf'\population_ACGs\other_mean{ext}',
        dpi=300,
        bbox_inches='tight'
        )


#%% heatmaps
fig, ax = plt.subplots(figsize=(1.8, 1.65))
ax.imshow(acgs_tagged, aspect='auto', interpolation='none', cmap='magma',
          extent=(-200, 200, 1, len(acgs_tagged)))
ax.set(title='tagged Dbh+',
       xlabel='lag (ms)',
       ylabel='cell #', yticks=(1,50))
for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\population_ACGs'
                rf'\tagged_ACGs{ext}',
                dpi=300,
                bbox_inches='tight')
plt.close(fig)
    
fig, ax = plt.subplots(figsize=(1.8, 1.65))
ax.imshow(acgs_putative, aspect='auto', interpolation='none', cmap='magma',
          extent=(-200, 200, 1, len(acgs_putative)))
ax.set(title='putative Dbh+',
       xlabel='lag (ms)',
       ylabel='cell #', yticks=(1,100,200))
for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\population_ACGs'
                rf'\putative_ACGs{ext}',
                dpi=300,
                bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(1.8, 1.65))
ax.imshow(acgs_other, aspect='auto', interpolation='none', cmap='magma',
          extent=(-200, 200, 1, len(acgs_other)))
ax.set(title='putative Dbh-',
       xlabel='lag (ms)',
       ylabel='cell #', yticks=(1,100))
for ext in ('.png', '.pdf'):
    fig.savefig(r'Z:\Dinghao\code_dinghao\LC_ephys\population_ACGs'
                rf'\other_ACGs{ext}',
                dpi=300,
                bbox_inches='tight')
plt.close(fig)