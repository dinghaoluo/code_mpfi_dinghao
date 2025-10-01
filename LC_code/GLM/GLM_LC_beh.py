# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:39:47 2025

GLM to try to tease apart the contributing factors to LC peak amplitude 
    under unmanipulated conditions (baseline and recovery trials in stim.
    sessions)

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import pickle 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import PerfectSeparationError

import matplotlib.pyplot as plt 
import seaborn as sns

from common import mpl_formatting
mpl_formatting()

import GLM_functions as gf

import rec_list
paths = rec_list.pathLC


#%% parameters 
SAMP_FREQ     = 1250  # Hz
SAMP_FREQ_BEH = 1000  # Hz
RUN_ONSET_IDX = 3 * SAMP_FREQ

eps = 1e-6

scaler = StandardScaler()


#%% path stems
all_sess_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/all_sessions')


#%% load cell table
print('loading data...')
cell_prop = pd.read_pickle('Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% main 
all_results = []
all_results_lr = []

for path in paths:
    recname = Path(path).name
    print(recname)
    
    # load behaviour
    beh_path = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC') / f'{recname}.pkl'
    try:
        with open(beh_path, 'rb') as f:
            beh = pickle.load(f)
    except Exception as e:
        print(f'beh file loading failed: {e}')
        
    timestamps_ms = beh['upsampled_timestamps_ms']
    timestamps_s  = timestamps_ms / 1000.0
    speeds_cm_s   = beh['upsampled_speed_cm_s']
    lick_times    = [lick[0] for trial in beh['lick_times'][1:] for lick in trial]
    cue_times     = beh['start_cue_times'][1:]
    reward_times  = beh['reward_times'][1:]
    run_onsets    = beh['run_onsets'][1:]
    trials        = beh['trial_statements'][1:]
    
    # find first opto trial
    first_opto_idx = next((i for i, t in enumerate(trials) if t[15] != '0'), None)
    if first_opto_idx is not None:
        trial_range = np.arange(first_opto_idx)
    else:
        trial_range = np.arange(len(trials))
        
    # design matrix 
    beh_rows, kept_trials = [], []
    start_time = run_onsets[0] / SAMP_FREQ_BEH
    
    for ti, onset in enumerate(run_onsets[:trial_range[-1]]):
        if np.isnan(onset):
            continue
    
        onset_time = onset / SAMP_FREQ_BEH
    
        # reward history
        reward_times_arr = np.array(reward_times) / SAMP_FREQ_BEH
        reward_feat = gf.time_since_last_reward(reward_times_arr, onset_time)
        if np.isnan(reward_feat):
            continue
    
        # lick rate last 5s
        lick_times_arr = np.array(lick_times) / SAMP_FREQ_BEH
        lick_rate = gf.lick_rate_last5s(lick_times_arr, onset_time)
        
        # time since start 
        time_since_start = onset_time - start_time
        
        # speed features
        mean_speed_trial = gf.mean_speed_last_trial(timestamps_s, speeds_cm_s,
                                                    [r/SAMP_FREQ_BEH for r in run_onsets], ti)
    
        feats = [reward_feat,
                 lick_rate,
                 time_since_start,
                 mean_speed_trial]
        beh_rows.append(feats)
        kept_trials.append(ti)
    
    X_behav_base = np.vstack(beh_rows)
    beh_names_base = (
        ['t. since last rew.',
         'lick rate last 5 s',
         't. since sess. start',
         'mean speed last trial']
    )
    
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
        for ti in kept_trials:
            amp = gf.run_onset_amplitude(trains[ti], SAMP_FREQ, RUN_ONSET_IDX)
            amp_rows.append(amp)
        amp_rows = np.array(amp_rows, dtype=float)
        
        # baseline amp
        base_rows = np.array([
            gf.baseline_rate(trains[ti], SAMP_FREQ, RUN_ONSET_IDX)
            for ti in kept_trials
        ], dtype=float)
        
        # make cell-specific X by adding prev_amp column
        prev_amp = np.concatenate([[np.nan], amp_rows[:-1]])
        X = np.column_stack([X_behav_base, base_rows, prev_amp])
        beh_names = beh_names_base + ['baseline rate', 'prev. run-onset amp.']
    
        # align with y
        y = np.clip(amp_rows, eps, None)
        row_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        row_mask &= ~(np.all(X == 0, axis=1))
        X = X[row_mask]
        y = y[row_mask]
    
        if len(y) < X.shape[1] + 2:
            print(f'{cluname}: too few valid trials; skipping')
            continue
        
        # colinearity check
        vif_data = pd.DataFrame()
        vif_data['feature'] = beh_names
        vif_data['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        
        # z-score
        Xz = scaler.fit_transform(X)
        
        # full model
        try:
            res_full = gf.fit_glm_log_gaussian(Xz, y, eps=eps)
        except PerfectSeparationError:
            print(f'{cluname}: full model failed')
            continue
        
        # back-prediction (in-sample)
        y_pred_log = res_full.predict(sm.add_constant(Xz))
        y_pred = np.exp(y_pred_log) - eps
        
        # correlation (Pearson’s r) and R²
        r_val = np.corrcoef(y, y_pred)[0, 1]
        r2_val = r_val**2
        
        # plot observed vs predicted amplitudes across trials
        plt.figure(figsize=(6,3))
        trials = np.arange(len(y))
        
        plt.scatter(trials, y, color='k', label='Observed', alpha=0.7)
        plt.scatter(trials, y_pred, color='red', label='Predicted', alpha=0.7)
        
        plt.xlabel('Trial index')
        plt.ylabel('Run-onset amplitude (Hz)')
        plt.title(f'{recname} - {cluname}\nR²={r2_val:.2f}, r={r_val:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # store coefficients (full model only)
        for name, coef in zip(beh_names, res_full.params[1:]):  # skip intercept
            all_results.append({
                'rec': recname,
                'cell_id': cluname,
                'predictor': name,
                'coef': coef
            })
        
        # drop-one-out LR tests
        for drop_name in beh_names:
            keep_idx = [i for i, n in enumerate(beh_names) if n != drop_name]
            X_reduced = Xz[:, keep_idx]
            try:
                res_reduced = gf.fit_glm_log_gaussian(X_reduced, y, eps=eps)
            except PerfectSeparationError:
                print(f'{cluname}: reduced model failed for {drop_name}')
                continue
            
            # compare fit
            llf_full = res_full.llf
            llf_red  = res_reduced.llf
            df_diff  = res_full.df_model - res_reduced.df_model
            lr_stat  = 2 * (llf_full - llf_red)
            p_lr     = stats.chi2.sf(lr_stat, df_diff)
            delta_aic = res_reduced.aic - res_full.aic
            
            # store
            all_results_lr.append({
                'rec': recname,
                'cell_id': cluname,
                'dropped': drop_name,
                'llf_full': llf_full,
                'llf_reduced': llf_red,
                'aic_full': res_full.aic,
                'aic_reduced': res_reduced.aic,
                'delta_aic': delta_aic,
                'lr_stat': lr_stat,
                'p_lr': p_lr
            })
    

results_df = pd.DataFrame(all_results)
lr_df = pd.DataFrame(all_results_lr)
    
    
#%% plotting 
coef_mean = results_df.groupby('predictor')['coef'].mean()
coef_sem  = results_df.groupby('predictor')['coef'].sem()

# run simple one-sample t-tests vs 0
stats_results = {}
for pred, group in results_df.groupby('predictor'):
    coefs = group['coef'].dropna().values
    if len(coefs) > 3:   # only if enough samples
        stat, p = wilcoxon(coefs, zero_method='pratt')
        stats_results[pred] = p
    else:
        stats_results[pred] = np.nan

# consistent order
order = [p for p in [
    't. since last rew.',
    'lick rate last 5 s',
    't. since sess. start',
    'mean speed last trial',
    'baseline rate',
    'prev. run-onset amp.'
] if p in coef_mean.index]

coef_mean = coef_mean.loc[order]
coef_sem  = coef_sem.loc[order]

plt.figure(figsize=(8,4))
ax = coef_mean.plot(kind='bar', yerr=coef_sem, capsize=3, color='lightgray', edgecolor='k')

# add significance stars
for i, pred in enumerate(coef_mean.index):
    p = stats_results.get(pred, np.nan)
    if np.isnan(p): 
        continue
    star = ''
    if p < 0.001: star = '***'
    elif p < 0.01: star = '**'
    elif p < 0.05: star = '*'
    if star:
        ax.text(i, coef_mean[pred] + coef_sem[pred] + 0.05, star,
                ha='center', va='bottom', color='k', fontsize=12)

plt.ylabel('log-amplitude coefficient (mean ± SEM)')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='k', lw=1)
plt.tight_layout()
plt.show()


#%% LR 
plt.figure(figsize=(7,4))
sns.boxplot(x='dropped', y='delta_aic', data=lr_df, color='lightgray')
plt.axhline(0, color='k', ls='--')
plt.ylabel('ΔAIC (reduced – full)')
plt.xlabel('Dropped predictor')
plt.title('Model improvement by predictor inclusion')
plt.xticks(rotation=45, ha='right')
plt.ylim(-5,5)
plt.tight_layout()
plt.show()