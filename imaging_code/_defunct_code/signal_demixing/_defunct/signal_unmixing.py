# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:00:00 2025

regress motion from both red and green, then unmix red from green and green 
    from red, for either the whole field or a single ROI

@author: Dinghao Luo
"""

#%% imports
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

sys.path.append(r'Z:\Dinghao\code_mpfi_dinghao\utils')
from common import mpl_formatting
mpl_formatting()


#%% core functions
def compute_motion_absdiff(R):
    """
    frame-wise absolute-diff motion regressor
    R: array shape (T, ...) or (T,), returns M of length T
    """
    R = np.asarray(R)
    T = R.shape[0]
    # flatten spatial dims if present
    diffs = np.abs(R[1:].reshape(T-1, -1) - R[:-1].reshape(T-1, -1))
    M = np.empty(T, dtype=np.float32)
    M[1:] = diffs.mean(axis=1)
    M[0] = M[1]
    return M


def compute_motion_corr(R):
    """
    negative Pearson-corr motion regressor
    R: array shape (T, ...) or (T,), returns M of length T
    """
    R = np.asarray(R)
    T = R.shape[0]
    M = np.empty(T, dtype=np.float32)
    for t in range(1, T):
        x = R[t-1].ravel() if R.ndim>1 else np.array([R[t-1]])
        y = R[t].ravel()   if R.ndim>1 else np.array([R[t]])
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(x, y)[0,1]
        M[t] = -0.0 if np.isnan(corr) else -corr
    M[0] = M[1]
    return M


def fit_regression(Y, X, robust=True):
    """
    fit Y = a * X + c via Huber or least-squares
    returns (a, c)
    """
    Y = np.asarray(Y)
    X = np.asarray(X)
    # mask NaNs
    mask = np.isfinite(X) & np.isfinite(Y)
    Xc = X[mask]
    Yc = Y[mask]
    A = np.vstack([Xc, np.ones_like(Xc)]).T
    if robust:
        model = HuberRegressor(fit_intercept=False)
        model.fit(A, Yc)
        a, c = model.coef_
    else:
        a, c = np.linalg.lstsq(A, Yc, rcond=None)[0]
    return a, c


#%% main function
def regress_and_unmix(bin_path, roi_idx=None):
    """
    regress motion from red and green; then cross-unmix both directions.

    Parameters:
    - red_bin: path to red channel .bin (data_chan2.bin)
    - roi_idx: None for whole-field, or integer ROI index for Suite2p F outputs

    Returns:
    - dict with raw, motion-cleaned, and final unmixed signals & coefficients
    """
    # brute force recname
    if 'AC' in bin_path:
        div = bin_path.find('-202')  # the most likely occurrence 
        recdate = bin_path[div-5 : div+9]  # full recdate 
        recsess = bin_path[div+10 : div+12]  # sess
        recname = f'{recdate}-{recsess}'
    else:
        recname = 'unknown'

    folder = os.path.dirname(bin_path)
    ops_path = os.path.join(folder, 'ops.npy')
    ops = np.load(ops_path, allow_pickle=True).item()
    T, H, W = ops['nframes'], ops['Ly'], ops['Lx']

    # load raw signals
    if roi_idx is None:
        mov_r = np.memmap(bin_path,   mode='r', dtype='int16', shape=(T,H,W)).astype(np.float32)
        mov_g = np.memmap(green_bin, mode='r', dtype='int16', shape=(T,H,W)).astype(np.float32)
        R_raw = mov_r.mean(axis=(1,2))
        G_raw = mov_g.mean(axis=(1,2))
        # compute motion regressors per channel
        M_R_abs  = compute_motion_absdiff(mov_r)
        M_G_abs  = compute_motion_absdiff(mov_g)
        M_R_corr = compute_motion_corr(mov_r)
        M_G_corr = compute_motion_corr(mov_g)
    else:
        F2 = np.load(os.path.join(folder, 'F_chan2.npy'))
        F1 = np.load(os.path.join(folder, 'F.npy'))
        R_raw = F2[:, roi_idx]
        G_raw = F1[:, roi_idx]
        M_R_abs  = compute_motion_absdiff(R_raw)
        M_G_abs  = compute_motion_absdiff(G_raw)
        M_R_corr = compute_motion_corr(R_raw)
        M_G_corr = compute_motion_corr(G_raw)

    #---- ABS-DIFF PATHWAY ----
    # motion regression per channel
    a_R_abs, c_R_abs = fit_regression(R_raw, M_R_abs)
    R_clean_abs = R_raw - a_R_abs * M_R_abs + c_R_abs
    a_G_abs, c_G_abs = fit_regression(G_raw, M_G_abs)
    G_clean_abs = G_raw - a_G_abs * M_G_abs + c_G_abs
    # cross-unmix
    b_RG_abs, d_RG_abs = fit_regression(G_clean_abs, R_clean_abs)
    G_final_abs = G_clean_abs - b_RG_abs * R_clean_abs + d_RG_abs
    b_GR_abs, d_GR_abs = fit_regression(R_clean_abs, G_clean_abs)
    R_final_abs = R_clean_abs - b_GR_abs * G_clean_abs + d_GR_abs

    #---- CORRELATION PATHWAY ----
    a_R_corr, c_R_corr = fit_regression(R_raw, M_R_corr)
    R_clean_corr = R_raw - a_R_corr * M_R_corr + c_R_corr
    a_G_corr, c_G_corr = fit_regression(G_raw, M_G_corr)
    G_clean_corr = G_raw - a_G_corr * M_G_corr + c_G_corr
    b_RG_corr, d_RG_corr = fit_regression(G_clean_corr, R_clean_corr)
    G_final_corr = G_clean_corr - b_RG_corr * R_clean_corr + d_RG_corr
    b_GR_corr, d_GR_corr = fit_regression(R_clean_corr, G_clean_corr)
    R_final_corr = R_clean_corr - b_GR_corr * G_clean_corr + d_GR_corr

    # collect results
    results = dict(
        R_raw=R_raw, G_raw=G_raw,
        M_R_abs=M_R_abs, M_G_abs=M_G_abs,
        M_R_corr=M_R_corr, M_G_corr=M_G_corr,
        # abs-diff
        a_R_abs=a_R_abs, c_R_abs=c_R_abs,
        a_G_abs=a_G_abs, c_G_abs=c_G_abs,
        b_RG_abs=b_RG_abs, d_RG_abs=d_RG_abs,
        b_GR_abs=b_GR_abs, d_GR_abs=d_GR_abs,
        R_clean_abs=R_clean_abs, G_clean_abs=G_clean_abs,
        R_final_abs=R_final_abs, G_final_abs=G_final_abs,
        # corr
        a_R_corr=a_R_corr, c_R_corr=c_R_corr,
        a_G_corr=a_G_corr, c_G_corr=c_G_corr,
        b_RG_corr=b_RG_corr, d_RG_corr=d_RG_corr,
        b_GR_corr=b_GR_corr, d_GR_corr=d_GR_corr,
        R_clean_corr=R_clean_corr, G_clean_corr=G_clean_corr,
        R_final_corr=R_final_corr, G_final_corr=G_final_corr,
        recname=recname
    )

    #---- plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(12,8), sharex=True)
    # abs-diff
    axes[0,0].plot(R_raw, 'b', label='raw R')
    axes[0,0].plot(R_final_abs, 'g', label='final R')
    axes[0,0].set_title('Red (abs-diff)')
    axes[0,0].legend()
    axes[1,0].plot(G_raw, 'm', label='raw G')
    axes[1,0].plot(G_final_abs, 'r', label='final G')
    axes[1,0].set_title('Green (abs-diff)')
    axes[1,0].legend()
    # corr
    axes[0,1].plot(R_raw, 'b')
    axes[0,1].plot(R_final_corr, 'g')
    axes[0,1].set_title('Red (corr)')
    axes[1,1].plot(G_raw, 'm')
    axes[1,1].plot(G_final_corr, 'r')
    axes[1,1].set_title('Green (corr)')
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Signal')
    axes[1,0].set_xlabel('Frame'); axes[1,1].set_xlabel('Frame')
    fig.suptitle(recname)
    plt.tight_layout(); plt.show()

    #---- save figure ----
    outdir = os.path.join(folder, 'unmix_results')
    os.makedirs(outdir, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(outdir, f'{recname}_unmixed{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    return results


#%% run
results = regress_and_unmix(
    bin_path=r'Z:\Jingyu\2P_Recording\AC976\AC976-20250424\06\RegOnly\suite2p\plane0\data_chan2.bin',
    roi_idx=None
    )