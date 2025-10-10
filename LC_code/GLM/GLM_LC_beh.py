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
LC_beh_stem = Path('Z:/Dinghao/code_dinghao/behaviour/all_experiments/LC')
GLM_stem = Path('Z:/Dinghao/code_dinghao/LC_ephys/GLM')


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
    beh_path = LC_beh_stem / f'{recname}.pkl'
    try:
        with open(beh_path, 'rb') as f:
            beh = pickle.load(f)
    except Exception as e:
        print(f'beh file loading failed: {e}')
        
    timestamps_ms = beh['upsampled_timestamps_ms']
    timestamps_s  = timestamps_ms / 1000.0
    speeds_cm_s   = beh['upsampled_speed_cm_s']
    lick_times    = [[lick[0] for lick in trial] for trial in beh['lick_times'][1:]]
    cue_times     = beh['start_cue_times'][1:]
    reward_times  = beh['reward_times'][1:]
    run_onsets    = beh['run_onsets'][1:]
    trials_sts    = beh['trial_statements'][1:]
    
    # find first opto trial
    opto_idx = [i for i, t in enumerate(trials_sts) if t[15] != '0']
    if not opto_idx:
        trial_range = np.arange(len(trials_sts))
        
    # design matrix 
    beh_rows, kept_trials = [], []
    start_time = run_onsets[0] / SAMP_FREQ_BEH
    
    for ti, onset in enumerate(run_onsets[:trial_range[-1]]):
        if ti in opto_idx or ti-1 in opto_idx or np.isnan(onset):
            continue
        onset_time = onset / SAMP_FREQ_BEH
        
        # first-lick-to-reward
        pred_err = gf.first_lick_to_reward_last_trial(lick_times, reward_times, ti)
        
        # time since last reward 
        time_since_rew = gf.time_since_last_reward(reward_times, onset_time, ti)
        
        # speed features
        mean_speed_trial = gf.mean_speed_curr_trial(timestamps_s, speeds_cm_s,
                                                    [r/SAMP_FREQ_BEH for r in run_onsets], ti)
        
        feats = [mean_speed_trial,
                 time_since_rew,
                 pred_err]
        beh_rows.append(feats)
        kept_trials.append(ti)
    
    X_behav_base = np.vstack(beh_rows)
    beh_names_base = (
        ['Mean speed',
         'Time since last reward',
         'Prediction error']
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
        beh_names = beh_names_base + ['Baseline FR', 'Prev. run-onset amp.']
    
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
        
        # plot single cell predictions vs observations 
        fig, ax = plt.subplots(figsize=(6, 3))
        trials = np.arange(len(y))
        
        for i, t in enumerate(trials):
            ax.plot([t, t], [y[i], y_pred[i]], color='gray', lw=0.6, alpha=0.5, zorder=0)
        
        ax.scatter(trials, y, color='k', s=20, label='Observed', alpha=0.7, zorder=2)
        ax.scatter(trials, y_pred, color='red', s=20, label='Predicted', alpha=0.7, zorder=3)
        
        ax.set(
            xlabel='Trial index',
            ylabel='Run-onset amplitude (Hz)',
            title=f'{cluname}\nR²={r2_val:.2f}, r={r_val:.2f}'
        )
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', length=3, width=0.8)
        ax.tick_params(axis='y', length=3, width=0.8)
        ax.legend(frameon=False, fontsize=8, loc='best')
        
        plt.tight_layout()
        for ext in ['.pdf', '.png']:
            fig.savefig(GLM_stem / f'single_cell_pred/{cluname}_pred_obs{ext}', 
                        dpi=300, 
                        bbox_inches='tight')
        plt.close()
        
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
            
            y_pred_log_red = res_reduced.predict(sm.add_constant(X_reduced))
            y_pred_red = np.exp(y_pred_log_red) - eps
            r_val_red = np.corrcoef(y, y_pred_red)[0,1]
            r2_red = r_val_red**2
            
            # compare fit
            delta_r2 = r2_val - r2_red
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
                'delta_r2': delta_r2,
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

stats_results = {}
for pred, group in results_df.groupby('predictor'):
    coefs = group['coef'].dropna().values
    if len(coefs) > 3:   # only if enough samples
        stat, p = wilcoxon(coefs, zero_method='pratt')
        stats_results[pred] = p
    else:
        stats_results[pred] = np.nan

# ordering from high to low 
order = coef_mean.sort_values(ascending=False).index.tolist()

coef_mean = coef_mean.loc[order]
coef_sem  = coef_sem.loc[order]

pvals = pd.Series(stats_results).loc[order]


#%% plotting 
fig, ax = plt.subplots(figsize=(4.5, 3))

ax.barh(order, coef_mean, xerr=coef_sem, color='#d9d9d9', edgecolor='k', capsize=2, height=0.6)

ax.axvline(0, color='k', lw=0.8)

for i, pred in enumerate(order):
    p = pvals[pred]
    if p < 0.0001 : star = '****'
    elif p < 0.001: star = '***'
    elif p < 0.01 : star = '**'
    elif p < 0.05 : star = '*'
    else          : star = ''
    if star:
        ax.text(coef_mean[pred] + np.sign(coef_mean[pred])*0.03,
                i, star, va='center', ha='center',
                fontsize=15, color='k')

ax.set(xlabel='Log-amplitude coefficient', 
       yticklabels=order,
       title='Predictors of LC run-onset amplitude')

ax.spines[['top','right']].set_visible(False)

for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'predictor_coeff{ext}',
                dpi=300,
                bbox_inches='tight')


#%% LR 
fig, ax = plt.subplots(figsize=(4,5))
sns.boxplot(x='dropped', y='delta_aic', data=lr_df, color='lightgray')
ax.axhline(0, color='k', ls='--')

ax.set(xlabel='Dropped predictor', 
       ylabel='ΔAIC (reduced – full)',
       title='Model AIC comparison')

plt.xticks(rotation=45, ha='right')

for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'model_AIC_comparison{ext}',
                dpi=300,
                bbox_inches='tight')


#%% ΔR² plot
mean_r2 = lr_df.groupby('dropped')['delta_r2'].mean().sort_values(ascending=False)
sem_r2  = lr_df.groupby('dropped')['delta_r2'].sem().loc[mean_r2.index]

fig, ax = plt.subplots(figsize=(4.5, 3))

ax.barh(mean_r2.index, mean_r2.values,
        xerr=sem_r2.values,
        color='#d9d9d9',
        edgecolor='k',
        capsize=2,
        height=0.6)

ax.axvline(0, color='k', lw=0.8)

ax.set(xlabel='unique variance explained (ΔR² ± SEM)',
       ylabel='',
       title='Drop-one-out variance explained by predictor')

ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(GLM_stem / f'deltaR2_by_predictor{ext}', 
                dpi=300, 
                bbox_inches='tight')