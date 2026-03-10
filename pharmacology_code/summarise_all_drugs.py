# -*- coding: utf-8 -*-
"""
summarise pharmacological experiments (Prazosin, Propranolol, SCH23390)

unified script

@author: Dinghao Luo
"""

#%% imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem 

from common_functions import mpl_formatting, replace_outlier, smooth_convolve
mpl_formatting()

from plotting_functions import plot_violin_with_scatter
import behaviour_functions as bf
import rec_list


#%% global parameters
pharmacology_stem = Path('Z:/Dinghao/code_dinghao/pharmacology')

tot_distance = 2200  # mm
XAXIS = np.arange(tot_distance) / 10  # cm


#%% core function
def summarise_drug(
        drug_name,
        paths,
        sessions,
        color,
        save_folder,
        sch_mode=False
    ):
    """
    sch_mode=False → baseline=session[0], drug=session[1]
    sch_mode=True  → baseline=session==1, drug=mean(session 2 & 3)
    """

    mean_speeds_baseline = []
    mean_speeds_drug     = []
    mean_licks_baseline  = []
    mean_licks_drug      = []
    reward_baseline      = []
    reward_drug          = []

    for pathname, sesslist in zip(paths, sessions):

        speed_drugs = []
        lick_drugs  = []
        reward_drugs = []

        for idx, sess in enumerate(sesslist):

            # -------- build txt path --------
            if sch_mode:
                sessname = Path(pathname).name
                recname = f'{sessname}-0{sess}'
                txtpath = rf'{pathname}\{recname}\{recname}T.txt'
            else:
                suffix = f'-0{sess}'
                txtpath = f'{pathname}{suffix}T.txt'

            file = bf.process_behavioural_data(txtpath)

            # -------- reward --------
            reward_times = file['reward_times'][1:-1]
            rewarded = [1 if not np.isnan(t) else 0 for t in reward_times]
            reward_val = np.mean(rewarded)

            # -------- speed --------
            speed_dist = np.array([
                replace_outlier(np.array(trial))
                for trial in file['speed_distances_aligned']
                if len(trial) > 0
            ])
            speed_mean = np.mean(speed_dist, axis=0)

            # prazosin/propranolol multiply by 1.8
            if not sch_mode:
                speed_mean *= 1.8

            # -------- licks --------
            lick_dist = np.array([
                smooth_convolve(np.array(trial), sigma=10) * 10
                for trial in file['lick_maps']
                if len(trial) > 0
            ])
            lick_mean = np.mean(lick_dist, axis=0)

            # -------- grouping logic --------
            if sch_mode:
                if sess == 1:
                    mean_speeds_baseline.append(speed_mean)
                    mean_licks_baseline.append(lick_mean)
                    reward_baseline.append(reward_val)
                elif sess in [2, 3]:
                    speed_drugs.append(speed_mean)
                    lick_drugs.append(lick_mean)
                    reward_drugs.append(reward_val)
            else:
                if idx == 0:
                    mean_speeds_baseline.append(speed_mean)
                    mean_licks_baseline.append(lick_mean)
                    reward_baseline.append(reward_val)
                elif idx == 1:
                    mean_speeds_drug.append(speed_mean)
                    mean_licks_drug.append(lick_mean)
                    reward_drug.append(reward_val)

        if sch_mode:
            mean_speeds_drug.append(np.mean(speed_drugs, axis=0))
            mean_licks_drug.append(np.mean(lick_drugs, axis=0))
            reward_drug.append(np.mean(reward_drugs))

    # convert to arrays
    mean_speeds_baseline = np.array(mean_speeds_baseline)
    mean_speeds_drug     = np.array(mean_speeds_drug)
    mean_licks_baseline  = np.array(mean_licks_baseline)
    mean_licks_drug      = np.array(mean_licks_drug)
    reward_baseline      = np.array(reward_baseline)
    reward_drug          = np.array(reward_drug)

    save_dir = pharmacology_stem / save_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # SPEED
    # =====================================================
    fig, ax = plt.subplots(figsize=(2, 1.7))

    ms_b = np.mean(mean_speeds_baseline, axis=0)
    ms_d = np.mean(mean_speeds_drug, axis=0)
    ss_b = sem(mean_speeds_baseline, axis=0)
    ss_d = sem(mean_speeds_drug, axis=0)

    ax.plot(XAXIS, ms_b, color='grey')
    ax.fill_between(XAXIS, ms_b+ss_b, ms_b-ss_b,
                    color='grey', alpha=.15, edgecolor='none')

    ax.plot(XAXIS, ms_d, color=color)
    ax.fill_between(XAXIS, ms_d+ss_d, ms_d-ss_d,
                    color=color, alpha=.15, edgecolor='none')

    ax.set(xlim=(0, 180), ylim=(0, 75),
           xlabel='Distance (cm)', ylabel='Speed (cm/s)')

    for s in ['top','right']:
        ax.spines[s].set_visible(False)

    fig.savefig(save_dir / 'speed_profile.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'speed_profile.pdf', dpi=300, bbox_inches='tight')

    # =====================================================
    # LICK
    # =====================================================
    fig, ax = plt.subplots(figsize=(2, 1.7))

    ml_b = np.mean(mean_licks_baseline, axis=0) / 10
    ml_d = np.mean(mean_licks_drug, axis=0) / 10
    sl_b = sem(mean_licks_baseline, axis=0) / 10
    sl_d = sem(mean_licks_drug, axis=0) / 10

    ax.plot(XAXIS, ml_b, color='grey')
    ax.fill_between(XAXIS, ml_b+sl_b, ml_b-sl_b,
                    color='grey', alpha=.15, edgecolor='none')

    ax.plot(XAXIS, ml_d, color=color)
    ax.fill_between(XAXIS, ml_d+sl_d, ml_d-sl_d,
                    color=color, alpha=.15, edgecolor='none')

    ax.set(xlim=(30, 219), ylim=(0, 0.6),
           xlabel='Distance (cm)', ylabel='Lick density (count/cm)')

    for s in ['top','right']:
        ax.spines[s].set_visible(False)

    fig.savefig(save_dir / 'lick_profile.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'lick_profile.pdf', dpi=300, bbox_inches='tight')

    # =====================================================
    # REWARD
    # =====================================================
    plot_violin_with_scatter(
        reward_baseline,
        reward_drug,
        'grey',
        color,
        figsize=(1.8,1.8),
        ylim=(-0.1,1.05),
        print_statistics=True,
        save=True,
        savepath=save_dir / 'reward_percentage'
    )
    
    return reward_drug


#%% RUN ALL DRUGS
reward_prazosin = summarise_drug(
    'Prazosin',
    rec_list.pathAlphaBlocker,
    rec_list.sessAlphaBlocker,
    color='darkgreen',
    save_folder='prazosin',
    sch_mode=False
)

reward_propranolol = summarise_drug(
    'Propranolol',
    rec_list.pathBetaBlocker,
    rec_list.sessBetaBlocker,
    color='darkcyan',
    save_folder='propranolol',
    sch_mode=False
)

reward_SCH = summarise_drug(
    'SCH23390',
    rec_list.pathSCH,
    rec_list.sessSCH,
    color='#004D80',
    save_folder='SCH23390',
    sch_mode=True
)


#%% compare rewards
fig, ax = plt.subplots(figsize=(2.2, 2.4))

drug_names = ['Prazosin', 'Propranolol', 'SCH23390']
colors = ['darkgreen', 'darkcyan', '#004D80']
data = [reward_prazosin, reward_propranolol, reward_SCH]

means = [np.mean(d) for d in data]
sems  = [sem(d) for d in data]

x = np.arange(3)

ax.bar(x, means, yerr=sems, capsize=2,
       color=colors, alpha=0.7, edgecolor='none')

# jittered scatter
for i, d in enumerate(data):
    jitter = np.random.uniform(-0.12, 0.12, size=len(d))
    ax.scatter(np.full(len(d), x[i]) + jitter,
               d,
               s=15,
               color='k',
               edgecolor='none',
               alpha=0.75)

ax.set(
    xticks=x,
    xticklabels=drug_names,
    ylabel='Reward percentage',
    ylim=(0, 1.035)
)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

for ext in ['.png', '.pdf']:
    fig.savefig(pharmacology_stem / f'reward_percentage_across_drugs{ext}',
                dpi=300, bbox_inches='tight')