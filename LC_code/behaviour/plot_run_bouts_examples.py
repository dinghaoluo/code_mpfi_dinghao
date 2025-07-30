# -*- coding: utf-8 -*-
"""
Created on Fri 27 June 17:16:12 2025

example plot: single session, targeted trial window

@author: Dinghao Luo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import mat73
import os
import sys

sys.path.append('Z:\\Dinghao\\code_mpfi_dinghao\\utils')
from common import mpl_formatting, smooth_convolve, gaussian_kernel_unity
mpl_formatting()


#%% parameters
SAMP_FREQ = 1250
gaus_speed = gaussian_kernel_unity(sigma=SAMP_FREQ*0.03)


#%% recname 
# note: t=30 will show trials 29-31

# recname = 'A067r-20230821-01'
# t = 30
# t = 157

recname = 'A045r-20221207-02'
t = 181  

# recname = 'A032r-20220802-02'
# t = 75


#%% paths
base_path = rf'Z:\Dinghao\MiceExp\ANMD{recname[1:5]}\{recname[:14]}\{recname}'

run_bout_path = rf'Z:\Dinghao\code_dinghao\run_bouts\{recname}_run_bouts_py.csv'
aligned_path = os.path.join(base_path, f'{recname}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat')
behave_lfp_path = os.path.join(base_path, f'{recname}_BehavElectrDataLFP.mat')
clu_path = os.path.join(base_path, f'{recname}.clu.1')
res_path = os.path.join(base_path, f'{recname}.res.1')
cell_profiles = pd.read_pickle(r'Z:/Dinghao/code_dinghao/LC_ephys/LC_all_cell_profiles.pkl')


#%% load data
run_bout_table = pd.read_csv(run_bout_path)
beh_lfp = mat73.loadmat(behave_lfp_path)
aligned = sio.loadmat(aligned_path)['trialsRun'][0][0]
tracks = beh_lfp['Track']
laps = beh_lfp['Laps']

clusters = np.loadtxt(clu_path, dtype=int, skiprows=1)
spike_times = np.loadtxt(res_path, dtype=int) / (20_000 / 1_250)
spike_times = spike_times.astype(int)

unique_clus = [clu for clu in np.unique(clusters) if clu not in [0,1]]
clu_to_row = {clu: i for i, clu in enumerate(unique_clus)}

max_time = spike_times.max() + 1
spike_map = np.zeros((len(unique_clus), max_time), dtype=int)
for time, clu in zip(spike_times, clusters):
    if clu in [0,1]:
        continue
    spike_map[clu_to_row[clu], time] = 1

spike_array = np.array([smooth_convolve(spike_map[i], sigma=int(SAMP_FREQ*.08))
                        for i in range(len(unique_clus))])

lickLfp = laps['lickLfpInd']
lickLfp_flat = []
for trial in range(len(lickLfp)):
    if isinstance(lickLfp[trial][0], np.ndarray):
        lickLfp_flat.extend(int(lick) for lick in lickLfp[trial][0])
lickLfp_flat = np.array(lickLfp_flat)

speed_MMsec = tracks['speed_MMsecAll']
speed_MMsec[speed_MMsec < 0] = np.nan
speed_MMsec = pd.Series(speed_MMsec).interpolate().fillna(method='bfill').fillna(method='ffill').values
speed_MMsec = np.convolve(speed_MMsec, gaus_speed, mode='same')/10

startLfpInd = aligned['startLfpInd'][0]
endLfpInd = aligned['endLfpInd'][0]


#%% select cells
selected_indices = []
for i, clu in enumerate(unique_clus):
    cluname = f'{recname} clu{clu}'
    if cluname in cell_profiles.index:
        profile = cell_profiles.loc[cluname]
        if profile['identity'] in ['tagged', 'putative'] and profile['run_onset_peak'] is True:
            selected_indices.append(i)

if not selected_indices:
    print(f'no tagged/putative run-onset peak cells found for {recname}')
    sys.exit()

print(f'{len(selected_indices)} tagged/putative run-onset peak cells selected.')


#%% plot trials
lfp_indices_t = np.arange(startLfpInd[t]-1250, min(endLfpInd[t+2]+1250, len(speed_MMsec)))
lap_start = lfp_indices_t[0]
xaxis = np.arange(0, len(lfp_indices_t)) / SAMP_FREQ

fig, ax = plt.subplots(figsize=(len(lfp_indices_t)/3000, 2.2))
ax.set(xlabel='time (s)', ylabel='speed (cm/s)',
       ylim=(0, 1.2 * max(speed_MMsec[lfp_indices_t])),
       xlim=(0, len(lfp_indices_t) / SAMP_FREQ),
       title=f'{recname} | mean tagged/putative | trials 29 to 31')
ax.plot(xaxis, speed_MMsec[lfp_indices_t], color='black')

startLfpInd_t = startLfpInd[np.in1d(startLfpInd, lfp_indices_t)]
ax.vlines((startLfpInd_t - lap_start)/SAMP_FREQ, 0, ax.get_ylim()[1], 'r', linestyle='dashed')

run_bout_t = run_bout_table.iloc[:,1][np.in1d(run_bout_table.iloc[:,1], lfp_indices_t)]
ax.vlines((run_bout_t - lap_start)/SAMP_FREQ, 0, ax.get_ylim()[1], 'g', linestyle='dashed')

ax_spk = ax.twinx()
ax_spk.set_ylabel('firing rate (Hz)', color='orange')
ax_spk.spines['right'].set_color('orange')
ax_spk.tick_params(axis='y', colors='orange', labelcolor='orange')

spike_subset = spike_array[selected_indices, :][:, lfp_indices_t] * SAMP_FREQ
mean_trace = np.mean(spike_subset, axis=0)
mean_trace = np.clip(mean_trace, 0, np.percentile(mean_trace, 99.5))
ax_spk.plot(xaxis, mean_trace, color='orange', linewidth=1.5)
ax_spk.set_ylim(0, np.max(mean_trace) * 1.1)

licks_t = lickLfp_flat[np.in1d(lickLfp_flat, lfp_indices_t)]
ax.vlines((licks_t - lap_start)/SAMP_FREQ, ax.get_ylim()[1], ax.get_ylim()[1] * 0.96, 'magenta')

ax.spines['top'].set_visible(False)
ax_spk.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()

for ext in ['.pdf', '.png']:
    fig.savefig(rf'Z:\Dinghao\paper\figures_other\{recname}_t{t}{ext}',
                dpi=300,
                bbox_inches='tight')