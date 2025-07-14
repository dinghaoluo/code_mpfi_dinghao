# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 2025

compare motion correction using reference (mean) frame regressor

mimics 'signal_unmixing.py' but computes M(t) = mean|R(x,y,t) - R_mean(x,y)|

@author: Dinghao Luo
"""

#%% imports
import numpy as np
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt


#%% functions
def compute_motion_absmean(R):
    """
    compute frame‐wise motion index via absolute difference
    from the mean reference frame.

    R: array (T, H, W)
    returns M (T,)
    """
    # reference image = long‐term mean
    ref = R.mean(axis=0)
    T = R.shape[0]
    # abs diff to mean
    diffs = np.abs(R - ref[None, :, :]).reshape(T, -1)
    M = diffs.mean(axis=1).astype(np.float32)
    return M

def compute_motion_corrmean(R):
    """
    compute frame‐wise motion index via negative Pearson correlation
    with the mean reference frame.

    R: array (T, H, W)
    returns M (T,)
    """
    ref = R.mean(axis=0).ravel()
    T = R.shape[0]
    M = np.empty(T, dtype=np.float32)
    for t in range(T):
        frame = R[t].ravel()
        corr = np.corrcoef(frame, ref)[0,1]
        M[t] = -corr
    return M

def fit_red_motion(R_ts, M, robust=True):
    """
    fit R_ts = a*M + c via Huber (or OLS) regression.
    returns (a, c)
    """
    X = np.vstack([M, np.ones_like(M)]).T
    if robust:
        model = HuberRegressor(fit_intercept=False)
        model.fit(X, R_ts)
        a, c = model.coef_
    else:
        a, c = np.linalg.lstsq(X, R_ts, rcond=None)[0]
    return a, c


#%% main 
ops = np.load(
    r'Z:\Jingyu\2P_Recording\AC967\AC967-20250213\04\rigid_reg\suite2p\plane0\ops.npy',
    allow_pickle=True
    ).item()
T, H, W = ops['nframes'], ops['Ly'], ops['Lx']

# load red‐channel movie
mov = np.memmap(
    r'Z:\Jingyu\2P_Recording\AC967\AC967-20250213\04\rigid_reg\suite2p\plane0\data_chan2.bin',
    mode='r', dtype='int16', shape=(T, H, W)
).astype(np.float32)

# mean red trace
R_ts = mov.mean(axis=(1,2))

# compute both motion regressors against mean‐frame
M_absmean  = compute_motion_absmean(mov)
M_corrmean = compute_motion_corrmean(mov)

# fit both
a_abs, c_abs   = fit_red_motion(R_ts, M_absmean,  robust=True)
a_corr, c_corr = fit_red_motion(R_ts, M_corrmean, robust=True)

print(f'Mean‐frame abs‐diff regressor:  a = {a_abs:.4f}, c = {c_abs:.4f}')
print(f'Mean‐frame corr regressor:      a = {a_corr:.4f}, c = {c_corr:.4f}')

# compute removed and corrected
removed_abs   = a_abs   * M_absmean
corrected_abs = R_ts     - a_abs   * M_absmean

removed_corr   = a_corr  * M_corrmean
corrected_corr = R_ts     - a_corr  * M_corrmean


#%% visualization
fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
axes[0].plot(R_ts,           color='tab:blue')
axes[0].set_ylabel('Original\nMean red')
axes[0].set_title('Original Mean Red Trace')
axes[1].plot(removed_abs,    color='tab:orange')
axes[1].set_ylabel('Removed\n(a·M_absmean)')
axes[1].set_title('Removed Components')
axes[2].plot(corrected_abs,  color='tab:green')
axes[2].set_ylabel('Corrected\nMean red')
axes[2].set_xlabel('Frame')
axes[2].set_title('Corrected Trace')
fig.suptitle('abs-diff-based correction')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


#%% visualization
fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
axes[0].plot(R_ts,            color='tab:blue')
axes[0].set_ylabel('Original\nMean red')
axes[0].set_title('Original Mean Red Trace')
axes[1].plot(removed_corr,    color='tab:orange')
axes[1].set_ylabel('Removed')
axes[1].set_title('Removed Component (corr‐to‐mean)')
axes[2].plot(corrected_corr,  color='tab:green')
axes[2].set_ylabel('Corrected\nMean red')
axes[2].set_xlabel('Frame')
axes[2].set_title('Corrected Trace')
fig.suptitle('mean-frame correlation-based correction')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()