# -*- coding: utf-8 -*-
"""
Created on Mon 7 July 2025

plot single cell rasters aligned to run, cue, and reward times,
overlaid with mean spike rate curves, across all sessions.

uses explicit time masking to plot only -1 to 4 s.

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import rec_list
paths = rec_list.pathLC

from common import mpl_formatting
mpl_formatting()

from common import colour_putative, colour_tagged


#%% parameters
SAMP_FREQ = 1250
BEF = 3
AFT = 7
MAX_LENGTH = BEF + AFT  # seconds
TIME_AXIS = np.arange(MAX_LENGTH * SAMP_FREQ) / SAMP_FREQ - BEF

# indices to show only -1 to 4 s
time_mask = (TIME_AXIS >= -1) & (TIME_AXIS <= 4)

# stems 
LC_ephys_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys')
all_sess_stem = LC_ephys_stem / 'all_sessions'


#%% load cell properties
cell_prop_path = LC_ephys_stem / 'LC_all_cell_profiles.pkl'
cell_prop = pd.read_pickle(cell_prop_path)
tag_list = [clu for clu in cell_prop.index if cell_prop['identity'][clu]=='tagged']
put_list = [clu for clu in cell_prop.index if cell_prop['identity'][clu]=='putative']


#%% loop over sessions
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    sess_folder = all_sess_stem / recname
    
    try:
        rasters_run = np.load(sess_folder / f'{recname}_all_rasters_run.npy',
                              allow_pickle=True).item()
        trains_run  = np.load(sess_folder / f'{recname}_all_trains_run.npy',
                              allow_pickle=True).item()
        rasters_cue = np.load(sess_folder / f'{recname}_all_rasters_cue.npy',
                              allow_pickle=True).item()
        trains_cue  = np.load(sess_folder / f'{recname}_all_trains_cue.npy',
                              allow_pickle=True).item()
        rasters_rew = np.load(sess_folder / f'{recname}_all_rasters_rew.npy',
                              allow_pickle=True).item()
        trains_rew  = np.load(sess_folder / f'{recname}_all_trains_rew.npy',
                              allow_pickle=True).item()
    except FileNotFoundError:
        print(f'{recname}: missing files; skipped')
        continue

    # get list of clusters
    clu_list = list(rasters_run.keys())

    for clu in tqdm(clu_list, total=len(clu_list)):
        suffix = ''
        if clu in tag_list: 
            suffix     = ' tgd'
            clu_colour = colour_tagged
        elif clu in put_list: 
            suffix     = ' put'
            clu_colour = colour_putative
        else:
            clu_colour = 'k'
        
        min_fr = min(
            np.nanmin(np.nanmean(trains_run[clu], axis=0)),
            np.nanmin(np.nanmean(trains_cue[clu], axis=0)),
            np.nanmin(np.nanmean(trains_rew[clu], axis=0))
            )
        max_fr = max(
            np.nanmax(np.nanmean(trains_run[clu], axis=0)),
            np.nanmax(np.nanmean(trains_cue[clu], axis=0)),
            np.nanmax(np.nanmean(trains_rew[clu], axis=0))
            )
        
        fig, axs = plt.subplots(1, 3, figsize=(6.8, 1.35))
        alignments = [
            ('Run', rasters_run, trains_run, 'Time from run onset (s)'),
            ('Cue', rasters_cue, trains_cue, 'Time from cue (s)'),
            ('Rew', rasters_rew, trains_rew, 'Time from reward (s)')
        ]
        
        for i, (align_name, rasters, trains, xlabel) in enumerate(alignments):
            ax = axs[i]
            axt = ax.twinx()
            ax.spines['top'].set_visible(False)
            axt.spines['top'].set_visible(False)
            
            raster = rasters[clu]
            train = trains[clu]
            tot_trial = raster.shape[0]
            
            # plot rasters
            for trial in range(tot_trial):
                spikes = np.where(raster[trial])[0] / SAMP_FREQ - BEF
                spikes_in_window = spikes[(spikes >= -1) & (spikes <= 4)]
                ax.scatter(spikes_in_window, np.full_like(spikes_in_window, trial+1),
                           color='grey', s=1, ec='none', alpha=.5)
            
            # plot mean firing rate using masking
            mean_profile_full = np.nanmean(train, axis=0)
            axt.plot(TIME_AXIS[time_mask], mean_profile_full[time_mask], color=clu_colour)
            
            ax.set(xlim=(-1,4), xlabel=xlabel, xticks=(0,2,4),
                   ylabel='Trial #', ylim=(-1, tot_trial+1))
            axt.set(ylabel='Firing rate\n(Hz)')
            
            # force y axis min 
            if min_fr < 2:
                axt.set_ylim(0, max_fr * 1.05)
            else:
                axt.set_ylim(min_fr * 0.99, max_fr * 1.05)
            
        
        fig.suptitle(f'{clu}{suffix}', fontsize=9)
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        
        outdir = LC_ephys_stem / 'single_cell_rasters' / 'all_alignments'
        outdir.mkdir(exist_ok=True)
        for ext in ('.png', '.pdf'):
            fig.savefig(
                outdir / f'{clu}{suffix}{ext}',
                dpi=300, bbox_inches='tight')
        plt.close()
