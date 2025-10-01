# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 10:23:47 2025

Test whether the ITI periods contain information about n-1 trial

@author: Dinghao Luo
"""

#%% imports 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
import statsmodels.api as sm


#%% helpers
def sec_to_idx(t_sec, fs):
    """convert seconds to integer sample index."""
    return int(np.round(t_sec * fs))

def window_indices(t0, t1, fs, n_timepoints):
    """return clipped [start, end) indices for a time window in samples."""
    a = max(0, sec_to_idx(t0, fs))
    b = min(n_timepoints, sec_to_idx(t1, fs))
    return a, max(a + 1, b)


#%% build trial tables
def compute_trial_features(trials, early_thresh=-0.3):
    """
    build a dataframe of per-trial behavior features.

    parameters:
    - trials: list of dicts with keys 'trial_index','trial_onset','cue_time',
      'reward_eligible_time','reward_time','lick_times'
    - early_thresh: float seconds; early-lick error is 'early' if first lick occurs
      at least this much before eligibility (negative value)

    returns:
    - df: pandas.DataFrame with columns:
        ['trial_index','trial_onset','cue_time','reward_eligible_time','reward_time',
         'first_lick_time','early_error','early_flag',
         'prev_early_error','prev_early_flag','next_lick_onset_time']
    """
    rows = []
    for tr in trials:
        licks = np.array(tr.get('lick_times', []), dtype=float)
        first_lick = np.nan if licks.size == 0 else float(licks.min())
        elig = tr['reward_eligible_time']
        early_err = np.nan if np.isnan(first_lick) else (elig - first_lick)
        early_flag = int((not np.isnan(early_err)) and (early_err < early_thresh))
        # next trial lick onset for outcome variable
        rows.append({
            'trial_index': tr['trial_index'],
            'trial_onset': tr['trial_onset'],
            'cue_time': tr['cue_time'],
            'reward_eligible_time': tr['reward_eligible_time'],
            'reward_time': tr.get('reward_time', np.nan),
            'first_lick_time': first_lick,
            'early_error': early_err,
            'early_flag': early_flag,
        })
    df = pd.DataFrame(rows).sort_values('trial_index').reset_index(drop=True)
    # add previous trial features
    df['prev_early_error'] = df['early_error'].shift(1)
    df['prev_early_flag'] = df['early_flag'].shift(1)
    # next-trial lick onset (dependent var for mediation)
    df['next_lick_onset_time'] = df['first_lick_time'].shift(-1)
    return df


#%% extract neural data 
def extract_onset_activity(F, fs, trials, onset_window=(0.0, 0.5), preclude_post_lick=True):
    """
    extract population activity vectors at next-trial onset windows.

    parameters:
    - F: np.ndarray (n_neurons, n_timepoints) dF/F or event rates
    - fs: float sampling rate in Hz
    - trials: list of dicts same as above
    - onset_window: tuple seconds relative to trial_onset, e.g., (0.0, 0.5)
    - preclude_post_lick: bool; if true, truncate window at first lick in that trial

    returns:
    - X_mean: np.ndarray (n_trials, n_neurons) mean activity over window
    - valid_mask: boolean array (n_trials,) true if window had >= 2 samples
    - win_bounds: list of (a,b) sample indices actually used per trial
    """
    n_neurons, n_time = F.shape
    X = []
    bounds = []
    valid = []
    for tr in trials:
        t0 = tr['trial_onset'] + onset_window[0]
        t1 = tr['trial_onset'] + onset_window[1]
        if preclude_post_lick and tr.get('lick_times', []):
            first_lick = np.min(tr['lick_times'])
            if first_lick < t1:
                t1 = max(t0, first_lick)  # truncate up to first lick
        a, b = window_indices(t0, t1, fs, n_time)
        if b - a < 2:
            X.append(np.full(n_neurons, np.nan))
            bounds.append((a, b))
            valid.append(False)
            continue
        x = np.nanmean(F[:, a:b], axis=1)
        X.append(x)
        bounds.append((a, b))
        valid.append(True)
    X_mean = np.vstack(X)
    return X_mean, np.array(valid, dtype=bool), bounds


#%% decoding with shuffle
def decode_prev_early(X, prev_flags, n_splits=5, random_state=0):
    """
    cross-validated decoding of (n-1 early_flag) from next-trial onset activity.

    parameters:
    - X: (n_trials, n_neurons) feature matrix (will z-score per fold)
    - prev_flags: (n_trials,) 0/1 labels for n-1 early vs not
    - n_splits: int stratified folds
    - random_state: int seed

    returns:
    - auc_cv: float mean roc-auc across folds
    - w_full: np.ndarray (n_neurons,) logistic weights fit on all valid trials
    """
    rs = check_random_state(random_state)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(prev_flags)
    Xv = X[mask]
    yv = prev_flags[mask].astype(int)

    if len(np.unique(yv)) < 2:
        return np.nan, np.full(X.shape[1], np.nan)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for tr_idx, te_idx in skf.split(Xv, yv):
        Xm = (Xv[tr_idx] - Xv[tr_idx].mean(0)) / (Xv[tr_idx].std(0) + 1e-8)
        Xe = (Xv[te_idx] - Xv[tr_idx].mean(0)) / (Xv[tr_idx].std(0) + 1e-8)
        clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
        clf.fit(Xm, yv[tr_idx])
        yhat = clf.predict_proba(Xe)[:, 1]
        aucs.append(roc_auc_score(yv[te_idx], yhat))
    auc_cv = float(np.mean(aucs))

    # fit on all valid trials to get the history axis
    Xs = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-8)
    clf_all = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
    clf_all.fit(Xs, yv)
    w = clf_all.coef_.ravel()
    return auc_cv, w

def shuffle_control_auc(X, prev_flags, n_shuffles=200, n_splits=5, random_state=0):
    """
    compute shuffle-control distribution of aucs.

    parameters:
    - X: feature matrix
    - prev_flags: labels
    - n_shuffles: number of label shuffles
    - n_splits: cv folds
    - random_state: seed

    returns:
    - auc_null: np.ndarray (n_shuffles,)
    """
    rs = check_random_state(random_state)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(prev_flags)
    Xv = X[mask]
    yv = prev_flags[mask].astype(int)
    if len(np.unique(yv)) < 2:
        return np.full(n_shuffles, np.nan)
    aucs = []
    for _ in range(n_shuffles):
        y_shuf = rs.permutation(yv)
        auc, _ = decode_prev_early(Xv, y_shuf, n_splits=n_splits, random_state=rs.randint(1e9))
        aucs.append(auc)
    return np.array(aucs)


#%% history axis 
def project_history_axis(X, w):
    """
    project trial activity onto the history axis.

    parameters:
    - X: (n_trials, n_neurons) feature matrix
    - w: (n_neurons,) logistic weight vector

    returns:
    - s: (n_trials,) projection scores (z-scored across valid trials)
    """
    mask_trials = np.isfinite(X).all(axis=1) & np.isfinite(w).all()
    s = np.full(X.shape[0], np.nan)
    if not np.any(mask_trials):
        return s
    s_raw = X[mask_trials] @ w
    s_z = (s_raw - np.nanmean(s_raw)) / (np.nanstd(s_raw) + 1e-8)
    s[mask_trials] = s_z
    return s

def simple_mediation(df, mediator_col='history_score',
                     x_col='prev_early_error', y_col='next_lick_onset_time'):
    """
    run a basic single-mediator model with ols (no bootstrapping).

    parameters:
    - df: pandas.DataFrame with columns for x, mediator, y
    - mediator_col: name of mediator column
    - x_col: name of predictor column
    - y_col: name of outcome column

    returns:
    - results: dict with keys 'a','b','c','c_prime','indirect','pvals'
    """
    d = df[[x_col, mediator_col, y_col]].dropna()
    if len(d) < 20:
        return {'a': np.nan, 'b': np.nan, 'c': np.nan, 'c_prime': np.nan,
                'indirect': np.nan, 'pvals': {}}

    Xc = sm.add_constant(d[[x_col]])
    # total effect c
    m_c = sm.OLS(d[y_col].values, Xc).fit()
    c = float(m_c.params[x_col])

    # path a
    m_a = sm.OLS(d[mediator_col].values, Xc).fit()
    a = float(m_a.params[x_col])

    # path b and c'
    Xcp = sm.add_constant(d[[x_col, mediator_col]])
    m_b = sm.OLS(d[y_col].values, Xcp).fit()
    b = float(m_b.params[mediator_col])
    c_prime = float(m_b.params[x_col])

    return {
        'a': a,
        'b': b,
        'c': c,
        'c_prime': c_prime,
        'indirect': a * b,
        'pvals': {
            'c': float(m_c.pvalues.get(x_col, np.nan)),
            'a': float(m_a.pvalues.get(x_col, np.nan)),
            'b_cprime': {k: float(v) for k, v in m_b.pvalues.items()}
        }
    }