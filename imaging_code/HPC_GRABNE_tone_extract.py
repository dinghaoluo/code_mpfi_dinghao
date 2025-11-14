# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:25:26 2025

extract GRABNE signals locked to tone activation 

@author: Dinghao Luo
"""

#%% imports 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 

import imaging_pipeline_functions as ipf
from common import mpl_formatting
mpl_formatting()

import rec_list
paths = rec_list.pathGRABNETone + rec_list.pathGRABNEToneDbhBlock


#%% path stems
BEF = 10
AFT = 30
SAMP_FREQ = 30  # Hz

all_sess_stem = Path('Z:/Dinghao/code_dinghao/GRABNE/all_sessions_tone')
mice_exp_stem = Path(r'Z:\Dinghao\MiceExp')


#%% helpers
def regress_out_motion(ch, ref, fit_mask=None, allow_lag=True, max_lag=5):
    """
    regress out motion carried by ref from ch using linear regression with optional small lag search.

    parameters:
    - ch: 1d ndarray, target signal (green dff)
    - ref: 1d ndarray, motion reference (red dff)
    - fit_mask: 1d boolean ndarray marking frames used to fit beta (True = use). if None, use all
    - allow_lag: bool, whether to search integer lag between channels
    - max_lag: int, maximum absolute lag in frames to try

    returns:
    - resid: 1d ndarray, ch with fitted motion component removed (residual)
    - beta: tuple of (intercept, slope) from best fit
    - best_lag: int, lag (in frames) applied to ref (positive means ref shifted forward)
    - r2: float, variance explained by motion on the fit_mask
    """
    ch = np.asarray(ch).astype(np.float64)
    ref = np.asarray(ref).astype(np.float64)
    n = ch.size

    if fit_mask is None:
        fit_mask = np.ones(n, dtype=bool)
    else:
        fit_mask = fit_mask.astype(bool)

    def fit_for_lag(lag):
        # shift ref by lag (np.roll), then zero out edges that wrap
        ref_shift = np.roll(ref, lag).copy()
        if lag > 0:
            ref_shift[:lag] = ref_shift[lag]  # avoid wrap artefact
        elif lag < 0:
            ref_shift[lag:] = ref_shift[lag-1]
        X = np.vstack([np.ones(n), ref_shift]).T
        m = fit_mask.copy()
        # ensure we don't use the padded edge points in the fit
        if lag > 0:
            m[:lag] = False
        elif lag < 0:
            m[lag:] = False
        if m.sum() < 5:
            return np.inf, (0.0, 0.0), None, -np.inf
        beta = np.linalg.lstsq(X[m], ch[m], rcond=None)[0]  # [intercept, slope]
        pred = X @ beta
        resid = ch - pred
        # variance explained on the mask
        r2 = 1.0 - np.var(resid[m]) / np.var(ch[m])
        mse = np.mean((ch[m] - pred[m])**2)
        return mse, (float(beta[0]), float(beta[1])), resid, r2

    best = (np.inf, (0.0, 0.0), None, -np.inf, 0)
    lags = [0]
    if allow_lag and max_lag > 0:
        lags = list(range(-max_lag, max_lag + 1))

    for lag in lags:
        mse, beta, resid, r2 = fit_for_lag(lag)
        if mse < best[0]:
            best = (mse, beta, resid, r2, lag)

    _, beta, resid, r2, best_lag = best
    return resid, beta, best_lag, float(r2)


#%% main 
for path in paths:
    recname = Path(path).name
    print(f'\n{recname}')
    
    plane_stem = Path(path) / 'suite2p/plane0'
    sessname = recname.replace('i', '')
    
    binpath = plane_stem / 'data.bin'
    bin2path = plane_stem / 'data_chan2.bin'
    opspath = plane_stem / 'ops.npy'
    txtpath = mice_exp_stem / f'ANMD{recname[1:4]}' / f'{sessname}T.txt'
    
    savepath = all_sess_stem / f'{recname}'
    savepath.mkdir(exist_ok=True)
    
    # load data 
    ops = np.load(opspath, allow_pickle=True).item()
    tot_frames = ops['nframes']
    shape = tot_frames, ops['Ly'], ops['Lx']
    
    print('loading movies and saving references...')
    mov  = np.memmap(binpath, mode='r', dtype='int16', shape=shape).astype(np.float32)
    mov2 = np.memmap(bin2path, mode='r', dtype='int16', shape=shape).astype(np.float32)
    
    tot_frames = mov.shape[0]  # once loaded, update tot_frames to be the max frame number
    
    ref  = ipf.plot_reference(mov, recname=recname, outpath=savepath, channel=1)
    ref2 = ipf.plot_reference(mov2, recname=recname, outpath=savepath, channel=2)
    
    raw_trace  = np.sum(mov, axis=(1,2))
    raw_trace2 = np.sum(mov2, axis=(1,2))
    
    dFF  = ipf.calculate_dFF(raw_trace)
    dFF2 = ipf.calculate_dFF(raw_trace2)
    
    # get tone stamps 
    txt = ipf.process_txt_nobeh(txtpath)
    frame_times   = txt['frame_times']
    buzzer_times  = txt['buzzer_times']
    buzzer_frames = [ipf.find_nearest(t, frame_times) for t in buzzer_times]

    # build fit mask: use all frames except tone-locked windows when fitting motion regression
    fit_mask = np.ones(tot_frames, dtype=bool)
    for buzz in buzzer_frames:
        lo = max(0, int(buzz - BEF * SAMP_FREQ))
        hi = min(tot_frames, int(buzz + AFT * SAMP_FREQ))
        fit_mask[lo:hi] = False

    # optionally also exclude frames with large suite2p shifts (if available)
    for key in ('xoff', 'yoff'):
        if key in ops:
            shifts = np.asarray(ops[key])
            thr = np.percentile(np.abs(shifts), 99.5)
            fit_mask[np.abs(shifts) > thr] = False

    # regress out motion from green using red as reference (search up to ±5-frame lag)
    dFF_corr, beta, lag, r2 = regress_out_motion(dFF, dFF2, fit_mask=fit_mask, allow_lag=True, max_lag=5)
    print(f'motion correction: beta0={beta[0]:.4f}, beta1={beta[1]:.4f}, lag={lag} frames, R²={r2:.3f}')

    # quick qc plot: raw green, raw red, corrected green
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(dFF, color='darkgreen', alpha=.6, label='green dFF')
    ax.plot(dFF2, color='darkred', alpha=.3, label='red dFF (motion)')
    ax.plot(dFF_corr, color='k', lw=1.0, label='green corrected')
    for buzz in buzzer_frames:
        ax.axvspan(buzz-1, buzz+1, alpha=.1)
    ax.legend(loc='upper right')
    ax.set_title(f'{recname}  (R² motion explained = {r2:.3f})')
    fig.tight_layout()
    fig.savefig(savepath / f'{recname}_qc_motion_correction.png', dpi=200)

    # align around tones
    tot_tones = len(buzzer_frames)
    win = (BEF + AFT) * SAMP_FREQ
    tone_aligned   = np.zeros((tot_tones, win))
    tone_aligned2  = np.zeros((tot_tones, win))
    tone_aligned_c = np.zeros((tot_tones, win))
    for i, buzz in enumerate(buzzer_frames):
        lo = int(buzz - BEF * SAMP_FREQ)
        hi = int(buzz + AFT * SAMP_FREQ)
        if lo >= 0 and hi < tot_frames:
            tone_aligned[i, :]   = dFF[lo:hi]
            tone_aligned2[i, :]  = dFF2[lo:hi]
            tone_aligned_c[i, :] = dFF_corr[lo:hi]
            
    # plot aligned means
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(tone_aligned.T, color='darkgreen', alpha=.02)
    ax.plot(np.mean(tone_aligned, axis=0), color='darkgreen', lw=1.2, label='green mean')
    ax.plot(np.mean(tone_aligned2, axis=0), color='darkred', alpha=.5, label='red mean')
    ax.plot(np.mean(tone_aligned_c, axis=0), color='k', lw=1.2, label='green corrected mean')
    ax.axvline(BEF * SAMP_FREQ, ls='--', lw=.8, color='k')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(savepath / f'{recname}_tone_aligned_means.png', dpi=200)
