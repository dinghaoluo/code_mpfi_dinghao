# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 10:37:03 2025

ITI decoding script to look at history dependency

@author: Dinghao Luo
"""

#%% imports 
import numpy as np
import pandas as pd
from history_dependency_functions import (
    compute_trial_features,
    extract_onset_activity,
    decode_prev_early,
    shuffle_control_auc,
    project_history_axis,
    simple_mediation
)

# =======================
# user-provided inputs
# =======================

# example placeholders, replace with your actual arrays
F = np.load('F.npy')  # shape (n_neurons, n_timepoints)
fs = 30.0             # Hz sampling rate
trials = np.load('trials.npy', allow_pickle=True).tolist()  # list of dicts

# parameters
onset_window = (0.0, 0.5)
early_thresh = -0.3
n_splits = 5
n_shuffles = 200
random_state = 0

# =======================
# step 1: build trial table
# =======================
df = compute_trial_features(trials, early_thresh=early_thresh)

# =======================
# step 2: extract next-trial onset activity
# =======================
X, valid, _ = extract_onset_activity(F, fs, trials,
                                     onset_window=onset_window,
                                     preclude_post_lick=True)
df['valid_feat'] = valid
df['prev_early_flag'] = df['prev_early_flag'].astype(float)

# =======================
# step 3: decoding (n–1 early vs not)
# =======================
auc, w = decode_prev_early(X, df['prev_early_flag'].values,
                           n_splits=n_splits,
                           random_state=random_state)
auc_null = shuffle_control_auc(X, df['prev_early_flag'].values,
                               n_shuffles=n_shuffles,
                               n_splits=n_splits,
                               random_state=random_state)

# =======================
# step 4: project history axis
# =======================
hist_score = project_history_axis(X, w)
df['history_score'] = hist_score

# =======================
# step 5: mediation
# =======================
med = simple_mediation(df,
                       mediator_col='history_score',
                       x_col='prev_early_error',
                       y_col='next_lick_onset_time')

# =======================
# outputs
# =======================
print('cv auc:', auc)
print('null auc mean±sd:', np.nanmean(auc_null), np.nanstd(auc_null))
print('mediation results:', med)

# optionally save dataframe + outputs
df.to_csv('trial_table_with_history.csv', index=False)
np.save('history_axis_weights.npy', w)
np.save('auc_null_distribution.npy', auc_null)
