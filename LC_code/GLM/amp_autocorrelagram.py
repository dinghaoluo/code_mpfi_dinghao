# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 17:01:02 2025

Addressing significant contributions of prev. run onset amp to curr onset amp
    found in GLM analysis 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import stats 

from common import mpl_formatting
import GLM_functions as gf
mpl_formatting()

import rec_list
paths = rec_list.pathLC


#%% path stems
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')
LC_beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
GLM_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM')


#%% parameters 
SAMP_FREQ     = 1250  # Hz
SAMP_FREQ_BEH = 1000  # Hz
RUN_ONSET_IDX = 3 * SAMP_FREQ

MAX_LAG = 10
N_SHUF = 50


#%% load cell table
print('loading data...')
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% main 
all_results      = []
all_results_shuf = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    # find the baseline trials 
    beh_path = LC_beh_stem / f'{recname}.pkl'
    try:
        with open(beh_path, 'rb') as f:
            beh = pickle.load(f)
    except Exception as e:
        print(f'beh file loading failed: {e}')
        
    trials_sts    = beh['trial_statements'][1:]
    
    # find first opto trial
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']
    if not opto_idx:
        trial_range = np.arange(len(trials_sts)-1)
    else:
        trial_range = np.arange(opto_idx[0])
    
    # single cell spiking data 
    all_trains_path = all_sess_stem / recname / f'{recname}_all_trains.npy'
    all_trains = np.load(all_trains_path, allow_pickle=True).item()
    
    curr_cell_prop = cell_prop[cell_prop['sessname']==recname]
    for cluname, row in curr_cell_prop.iterrows():
        if row['identity'] == 'other' or not row['run_onset_peak']:
            continue
    
        trains = all_trains[cluname]
    
        # compute amplitudes
        amp_rows = []
        for ti in trial_range:
            amp = gf.run_onset_amplitude(trains[ti], SAMP_FREQ, RUN_ONSET_IDX)
            amp_rows.append(amp)
        
        amp_rows = np.array(amp_rows, dtype=float)
        
        # compute acg up to 10-trial lag
        acg_vals      = []
        acg_shuf_vals = []
        for lag in range(1, MAX_LAG + 1):
            r, p = stats.pearsonr(amp_rows[:-lag], amp_rows[lag:])
            acg_vals.append(r)
            all_results.append({
                'rec': recname,
                'cluname': cluname,
                'lag': lag,
                'acg': r,
                'p_acg': p
            })
            
            # shuffled ACG (mean of 50 shuffles)
            r_shufs = []
            for _ in range(N_SHUF):
                amp_rows_shuf = amp_rows.copy()
                np.random.shuffle(amp_rows_shuf)
                
                r_shufs.append(np.corrcoef(amp_rows_shuf[:-lag], amp_rows_shuf[lag:])[0, 1])
            
            r_shuf_mean = np.mean(r_shufs)

            acg_shuf_vals.append(r_shuf_mean)
            all_results_shuf.append({
                'rec': recname,
                'cluname': cluname,
                'lag': lag,
                'acg': r_shuf_mean
            })
            
            
#%% plotting 
acg_df      = pd.DataFrame(all_results)
acg_shuf_df = pd.DataFrame(all_results_shuf)

mean_acg      = acg_df.groupby('lag')['acg'].mean()
sem_acg       = acg_df.groupby('lag')['acg'].sem()
mean_acg_shuf = acg_shuf_df.groupby('lag')['acg'].mean()
sem_acg_shuf  = acg_shuf_df.groupby('lag')['acg'].sem()

lags = sorted(acg_df['lag'].unique())
p_lag = []

for lag in lags:
    # collect per-cell r-values at this lag
    r_vals = acg_df.loc[acg_df['lag'] == lag, 'acg'].dropna()
    if len(r_vals) > 3:
        # two-sided test vs zero correlation
        stat, p = stats.wilcoxon(r_vals, zero_method='pratt', alternative='two-sided')
    else:
        p = np.nan
    p_lag.append(p)

p_lag = np.array(p_lag)
sig_mask = p_lag < 0.05

print('Per-lag Wilcoxon p-values:')
for lag, p in zip(lags, p_lag):
    print(f'  lag {lag}: p = {p:.4f}')

fig, ax = plt.subplots(figsize=(2.6, 2.4))
ax.errorbar(mean_acg.index, mean_acg.values, yerr=sem_acg.values,
            fmt='-o', lw=1.2, ms=3, color='k', capsize=2, label='real')
ax.axhline(0, color='grey', ls='--', lw=0.75)

# annotate p-values and mean ± sem
for lag, y, sem, p in zip(lags, mean_acg.loc[lags], sem_acg.loc[lags], p_lag):
    ax.text(lag, y + 0.025, f'p={p:.5f}\n{y:.4f}±{sem:.4f}',
            ha='center', va='bottom', fontsize=3)

ax.set(
    xlabel='Lag (trial)',
    ylabel='Autocorrelation (r)',
    title='Run-onset amplitude ACG'
)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='x', length=3, width=0.8)
ax.tick_params(axis='y', length=3, width=0.8)
plt.tight_layout()

for ext in ('.pdf', '.png'):
    fig.savefig(GLM_stem / f'trial_lag_autocorrelation{ext}',
                dpi=300, bbox_inches='tight')