# -*- coding: utf-8 -*-
"""
Created on Sat 5 Aug 14:23:46 2023

theta analysis aligned to LC stim

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 
import mat73
import sys 
from scipy.stats import circvar, wilcoxon, sem
import math

if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% functions
def conv_db(x):
    if isinstance(x, np.ndarray):
        return [20*math.log10(i) for i in x]
    else:
        return 20*math.log10(x)
    

def cir_shuf(x, shuf=500):
    "input a vector and thou shall get a shuffled vector.--God"
    length = len(x)
    shuf_array = np.zeros((500, length))
    rand_shift = np.random.randint(1, length/2, shuf)
    
    for i, shift in enumerate(rand_shift):
        shuf_array[i,:] = np.roll(x, -shift)
    
    return [np.mean(shuf_array, axis=0),
            np.percentile(shuf_array, [5, 95], axis=0, interpolation='midpoint')]
    
    
def sum_vector(heights):
    thetas = np.linspace(-3.14, 3.14, 13)
    
    x_sum = 0
    y_sum = 0
    
    for i, height in enumerate(heights):
        x = height*math.cos(thetas[i])
        y = height*math.sin(thetas[i])
        x_sum+=x
        y_sum+=y
    
    r_sum = math.sqrt(x_sum**2 + y_sum**2)
    theta_sum = math.atan2(y_sum, x_sum)
    
    return [r_sum, theta_sum]
    

def plot_polar_phase(recname, peri_stims_theta_phase, peri_start_theta_phase, peri_cont_theta_phase):
    all_stims_theta_phase = [s[1250+125] for s in peri_stims_theta_phase]
    all_cont_theta_phase = [s[1250+125] for s in peri_cont_theta_phase]
    all_start_theta_phase = [s[1250+125] for s in peri_start_theta_phase]

    nbins = 12; width = (2*np.pi) / nbins
    histogram_stims = np.histogram(all_stims_theta_phase, bins=nbins, range=(-3.14, 3.14))
    histogram_cont = np.histogram(all_cont_theta_phase, bins=nbins, range=(-3.14, 3.14))
    histogram_start = np.histogram(all_start_theta_phase, bins=nbins, range=(-3.14, 3.14))

    fig, axs = plt.subplots(1,2,figsize=(6,3), subplot_kw={'projection': 'polar'})

    theta_cont = histogram_cont[1][:-1]
    radii_cont = histogram_cont[0]
    bars_cont = axs[0].bar(theta_cont, radii_cont, width=width, edgecolor='k')
    rv_cont = sum_vector(radii_cont)
    axs[0].arrow(rv_cont[1], 0, 0, rv_cont[0], 
                 length_includes_head=True, head_width=0.5, head_length=3,
                 facecolor='r', edgecolor='r')

    theta_stims = histogram_stims[1][:-1]
    radii_stims = histogram_stims[0]
    bars_stims = axs[1].bar(theta_stims, radii_stims, width=width, edgecolor='k')
    rv_stims = sum_vector(radii_stims)
    axs[1].arrow(rv_stims[1], 0, 0, rv_stims[0],
                 length_includes_head=True, head_width=0.1, head_length=2,
                 facecolor='r', edgecolor='r')

    # Use custom colors and opacity
    for r, bar in zip(radii_cont, bars_cont):
        bar.set_facecolor(plt.cm.jet(r / 30.))
        bar.set_alpha(0.8)
    for r, bar in zip(radii_stims, bars_stims):
        bar.set_facecolor(plt.cm.jet(r / 30.))
        bar.set_alpha(0.8)
        
    for i in range(2):
        axs[i].set_rticks([0,5,10,20,30])
        axs[i].set_yticklabels(['','',10,20,30])
        axs[i].set_rlabel_position(60)

    axs[0].set(title='stim control')
    axs[1].set(title='stimulation')

    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\{}\phase_stim_stimcont_{}.png'.format(recname, recname),
                bbox_inches='tight',
                dpi=500)
    
    plt.close(fig)
    
    
    # stim v stim cont
    fig, axs = plt.subplots(1,2,figsize=(6,3), subplot_kw={'projection': 'polar'})

    theta_start = histogram_start[1][:-1]
    radii_start = histogram_start[0]
    
    bars_start = axs[0].bar(theta_start, radii_start, width=width, edgecolor='k')
    bars_stims = axs[1].bar(theta_stims, radii_stims, width=width, edgecolor='k')
    
    rv_start = sum_vector(radii_start)
    axs[0].arrow(rv_start[1], 0, 0, rv_start[0],
                 length_includes_head=True, head_width=0.25, head_length=0.5,
                 facecolor='r', edgecolor='r')

    rv_stims = sum_vector(radii_stims)
    axs[1].arrow(rv_stims[1], 0, 0, rv_stims[0],
                 length_includes_head=True, head_width=0.25, head_length=0.5,
                 facecolor='r', edgecolor='r')

    # Use custom colors and opacity
    for r, bar in zip(radii_start, bars_start):
        bar.set_facecolor(plt.cm.jet(r / 30.))
        bar.set_alpha(0.8)
    for r, bar in zip(radii_stims, bars_stims):
        bar.set_facecolor(plt.cm.jet(r / 30.))
        bar.set_alpha(0.8)
        
    for i in range(2):
        axs[i].set_rticks([0,5,10,20,30])
        axs[i].set_yticklabels(['','',10,20,30])
        axs[i].set_rlabel_position(60)

    axs[0].set(title='baseline')
    axs[1].set(title='stimulation')

    fig.tight_layout()
    plt.show()
    
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\{}\phase_stim_baseline_{}.png'.format(recname, recname),
                bbox_inches='tight',
                dpi=500)
    
    plt.close(fig)
    
    
    # # simple histogram as demo 
    # fig, ax = plt.subplots(figsize=(5,2))
    
    # ax.bar(histogram_stims[1][:-1], histogram_stims[0], width=width, edgecolor='k')
    
    # ax.set(xlabel='cir. dev. (π)', ylabel='frequency')
    
    # fig.tight_layout()
    # plt.show()
    
    # fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\{}\phase_stim_{}_hist.png'.format(recname, recname),
    #             bbox_inches='tight',
    #             dpi=500)

    # plt.close(fig)


#%% plotting parameters 
import matplotlib
taxis = np.arange(-1250, 6250)/1250
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% main 
all_stim_theta_amplitude = []; all_stims_theta_amplitude = []
all_start_theta_amplitude = []; all_cont_theta_amplitude = []; all_recov_theta_amplitude = []

all_stim_theta_frequency = []; all_stims_theta_frequency = []
all_start_theta_frequency = []; all_cont_theta_frequency = []; all_recov_theta_frequency = []

all_stims_theta_phase = []
all_start_theta_phase = []; all_cont_theta_phase = []; all_recov_theta_phase = []

all_stim_cir_dev = []; all_stims_cir_dev = []
all_start_cir_dev = []; all_cont_cir_dev = []; all_recov_cir_dev = []

shuf_all_stims_cir_dev = []; shuf_all_stims_cir_dev_perc = []

for sessname in pathHPC:
    recname = sessname[43:60]  # recording name, e.g. A069r-20230905-01
    print(recname)
    
    theta = mat73.loadmat('{}/{}_eeg_1250Hz.mat'.format(sessname, recname))

    try: 
        theta_h = theta['ThetaPhase_hilbert'][:,0]
        theta_amp = theta['ThetaAmp_hilbert'][:,0]
        theta_freq = theta['ThetaFreq_hilbert'][:,0]
    except IndexError:  # when there is only one shank
        theta_h = theta['ThetaPhase_hilbert'][:]
        theta_amp = theta['ThetaAmp_hilbert'][:]
        theta_freq = theta['ThetaFreq_hilbert'][:]

    mat_BTDT = sio.loadmat('{}/{}BTDT.mat'.format(sessname, recname))
    behEvents = mat_BTDT['behEventsTdt']
    stim_tps = behEvents['stimPulse'][0,0][:,1]

    stims = []
    last_stim = stim_tps[0]; stims.append(last_stim)
    for t in stim_tps[1:]:
        if t-last_stim < 1250:  # 1 second within the last stim 
            pass
        else:
            last_stim = t
            stims.append(t)
        
    alignRun = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(sessname, recname))
    # # alignCue = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat'.format(sessname, recname))
    behInfo = sio.loadmat('{}/{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(sessname, recname))['beh']
    stim_trial = np.squeeze(np.where(behInfo['pulseMethod'][0][0][0]!=0))-1  # -1 to match up with matlab indexing
    stim_cont = stim_trial+2
    tot_trial = len(behInfo['pulseMethod'][0][0][0])
    
    # cues = alignCue['trialsCue']['startLfpInd'][0][0][0]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]

    # data wrangling
    peri_stim_theta_amp = []
    peri_stim_theta_phase = []
    peri_stim_theta_freq = []
    peri_stims_theta_amp = []
    peri_stims_theta_phase = []
    peri_stims_theta_freq = []
    
    shuf_stims_theta_phase = []
    
    for t in stim_trial:
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # -1~5 s 
        peri_stim_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_stim_theta_phase.append(phase_seq)
        
        freq_seq = theta_freq[s-1250:s+6250]
        peri_stim_theta_freq.append(freq_seq)
        
    for t in stims:
        amp_seq = theta_amp[t-1250:t+6250]  # -1~5 s 
        peri_stims_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[t-1250:t+6250]
        peri_stims_theta_phase.append(phase_seq)
        
        freq_seq = theta_freq[t-1250:t+6250]
        peri_stims_theta_freq.append(freq_seq)
        
    avg_peri_stim_theta_amplitude = np.mean(peri_stim_theta_amp, axis=0)
    avg_peri_stims_theta_amplitude = np.mean(peri_stims_theta_amp, axis=0)
    all_stim_theta_amplitude.append(avg_peri_stim_theta_amplitude)
    all_stims_theta_amplitude.append(avg_peri_stims_theta_amplitude)
    
    peri_stim_cir_dev = circvar(peri_stim_theta_phase, high=3.14159, low =-3.14159,
                                axis=0)
    all_stim_cir_dev.append(peri_stim_cir_dev)
    
    all_stim_theta_frequency.append(peri_stim_theta_freq)
    
    all_stims_theta_phase.append(peri_stims_theta_phase)
    peri_stims_cir_dev = circvar(peri_stims_theta_phase, high=3.14159, low =-3.14159,
                                axis=0)
    all_stims_cir_dev.append(peri_stims_cir_dev)
    
    [shuf_stims_cir_dev, shuf_stims_cir_dev_perc] = cir_shuf(peri_stims_cir_dev)
    shuf_all_stims_cir_dev.append(shuf_stims_cir_dev)
    shuf_all_stims_cir_dev_perc.append(shuf_stims_cir_dev_perc)
    
    all_stims_theta_frequency.append(peri_stim_theta_freq)
    
    
    peri_start_theta_amp = []
    peri_start_theta_phase = []
    peri_start_theta_freq = []
    for t in np.arange(stim_trial[0]):
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # 3 s around 
        peri_start_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_start_theta_phase.append(phase_seq)
        
        freq_seq = theta_freq[s-1250:s+6250]
        peri_start_theta_freq.append(freq_seq)
        
    avg_peri_start_theta_amplitude = np.mean(peri_start_theta_amp, axis=0)
    all_start_theta_amplitude.append(avg_peri_start_theta_amplitude)
        
    all_start_theta_phase.append(peri_start_theta_phase)
    peri_start_cir_dev = circvar(peri_start_theta_phase, high=3.14159, low =-3.14159,
                                 axis=0)
    all_start_cir_dev.append(peri_start_cir_dev)
    
    all_start_theta_frequency.append(peri_start_theta_freq)
    
    
    peri_cont_theta_amp = []
    peri_cont_theta_phase = []
    peri_cont_theta_freq = []
    for t in stim_cont:
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # 3 s around 
        peri_cont_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_cont_theta_phase.append(phase_seq)
        
        freq_seq = theta_freq[s-1250:s+6250]
        peri_cont_theta_freq.append(freq_seq)
        
    avg_peri_cont_theta_amplitude = np.mean(peri_cont_theta_amp, axis=0)
    all_cont_theta_amplitude.append(avg_peri_cont_theta_amplitude)
        
    all_cont_theta_phase.append(peri_cont_theta_phase)
    peri_cont_cir_dev = circvar(peri_cont_theta_phase, high=3.14159, low =-3.14159,
                                axis=0)
    all_cont_cir_dev.append(peri_cont_cir_dev)
    
    all_cont_theta_frequency.append(peri_start_theta_freq)
    
    
    peri_recov_theta_amp = []
    peri_recov_theta_phase = []
    for t in np.arange(stim_trial[-1],tot_trial-1):
        s = starts[t]
        amp_seq = theta_amp[s-1250:s+6250]  # 3 s around 
        peri_recov_theta_amp.append(amp_seq)
        
        phase_seq = theta_h[s-1250:s+6250]
        peri_recov_theta_phase.append(phase_seq)
    avg_peri_recov_theta_amplitude = np.mean(peri_recov_theta_amp, axis=0)
    all_recov_theta_amplitude.append(avg_peri_recov_theta_amplitude)
        
    all_recov_theta_phase.append(peri_recov_theta_phase)
    peri_recov_cir_dev = circvar(peri_recov_theta_phase, high=3.14159, low =-3.14159,
                                 axis=0)
    all_recov_cir_dev.append(peri_recov_cir_dev)
    
    
    # session by session plotting 
    plot_polar_phase(recname, peri_stims_theta_phase, peri_start_theta_phase, peri_cont_theta_phase)


#%% temp plotting cell (eeg trace)
# eeg = theta['eeg'][:,0]

# fig, ax = plt.subplots(figsize=(6,1))

# for p in ['bottom','top','left','right']:
#     ax.spines[p].set_visible(False)
# ax.set(yticks=[], xticks=[])

# ax.plot(eeg[50000:55000], c='k', linewidth=1)  # 4s

# fig.tight_layout()
# plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\eg_eeg.png',
#             bbox_inches='tight',
#             dpi=500)

# plt.close(fig)


#%% temp plotting cell (eg theta)
# fig, ax = plt.subplots(figsize=(6,1))
# for p in ['bottom','top','left','right']:
#     ax.spines[p].set_visible(False)
# ax.set(yticks=[], xticks=[])

# xaxis = np.arange(5000)/100

# y = np.sin(xaxis)

# ax.plot(xaxis, y, c='r', linewidth=1)

# fig.tight_layout
# plt.show()

# fig.savefig('Z:\Dinghao\code_dinghao\LC_figures\eg_theta.png',
#             bbox_inches='tight',
#             dpi=500)

# plt.close(fig)



#%% plot theta amplitude (stim v baseline)
fig, ax = plt.subplots(figsize=(3.5,3))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)
ax.set(ylabel='norm. theta amplitude', xlabel='time (s)',
       yticks=[0,.5,1])
fig.suptitle('theta amplitude, stim v baseline')

mean_amp_stim = np.mean(all_stim_theta_amplitude, axis=0)
mean_amp_start = np.mean(all_start_theta_amplitude, axis=0)
mean_amp_max = max(max(mean_amp_stim), max(mean_amp_start))
mean_amp_min = min(min(mean_amp_stim), min(mean_amp_start))

# min max normalisation
mean_amp_stim = [(s-mean_amp_min)/(mean_amp_max-mean_amp_min) for s in mean_amp_stim]
mean_amp_start = [(s-mean_amp_min)/(mean_amp_max-mean_amp_min) for s in mean_amp_start]

ax.plot(taxis, mean_amp_stim, 'royalblue')
ax.plot(taxis, mean_amp_start, 'grey', alpha=.5)

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_baseline_theta_amp.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


#%% calculate and plot comparison theta amplitude (stim v baseline)
amp_stims = []; amp_starts = []

for sess in all_stim_theta_amplitude:
    amp_stims.append(np.mean(sess[2500:3750]))  # 1~2 s
for sess in all_start_theta_amplitude:
    amp_starts.append(np.mean(sess[2500:3750]))
amp_max = max(max(amp_stims), max(amp_starts))
amp_min = min(min(amp_stims), min(amp_starts))

# min max normalisation
amp_stims = [(s-amp_min)/(amp_max-amp_min) for s in amp_stims]
amp_starts = [(s-amp_min)/(amp_max-amp_min) for s in amp_starts]

pval = wilcoxon(amp_stims, amp_starts)[1]


fig, ax = plt.subplots(figsize=(3,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.set_xticklabels(['non-stim', 'stim'], minor=False)
ax.set_xticks([1,2])
ax.set_yticks([0,.5,1])

ax.set(ylabel='norm. theta amplitude')
fig.suptitle('theta amplitude, stim v baseline, p={}'.format(round(pval, 3)))

bp = ax.bar([1, 2], [np.mean(amp_starts), np.mean(amp_stims)],
            color=['grey', 'royalblue'], edgecolor=['k','k'], width=.35)
    
ax.scatter([[1]*len(amp_starts), [2]*len(amp_stims)], [amp_starts, amp_stims], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[1]*len(amp_starts), [2]*len(amp_stims)], [amp_starts, amp_stims], zorder=2,
        color='grey', alpha=.5)

ax.set(xlim=(0.5, 2.5), ylim=(-.01, 1.01))

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_baseline_theta_amp_bar.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


#%% plot theta amplitude (stim v stim cont)
fig, ax = plt.subplots(figsize=(3.5,3))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)
ax.set(ylabel='norm. theta amplitude', xlabel='time (s)',
       yticks=[0,.5,1])
fig.suptitle('theta amplitude, stim v stim-cont')

mean_amp_stim = np.mean(all_stim_theta_amplitude, axis=0)
mean_amp_cont = np.mean(all_cont_theta_amplitude, axis=0)
mean_amp_max = max(max(mean_amp_stim), max(mean_amp_cont))
mean_amp_min = min(min(mean_amp_stim), min(mean_amp_cont))

# min max normalisation
mean_amp_stim = [(s-mean_amp_min)/(mean_amp_max-mean_amp_min) for s in mean_amp_stim]
mean_amp_cont = [(s-mean_amp_min)/(mean_amp_max-mean_amp_min) for s in mean_amp_cont]

ax.plot(taxis, mean_amp_stim, 'royalblue')
ax.plot(taxis, mean_amp_cont, 'grey', alpha=.5)

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_theta_amp.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


#%% calculate and plot comparison theta amplitude (stim v stim cont)
amp_stims = []; amp_conts = []

for sess in all_stim_theta_amplitude:
    amp_stims.append(np.mean(sess[2500:3750]))  # 1~2 s
for sess in all_cont_theta_amplitude:
    amp_conts.append(np.mean(sess[2500:3750]))
amp_max = max(max(amp_stims), max(amp_conts))
amp_min = min(min(amp_stims), min(amp_conts))

# min max normalisation
amp_stims = [(s-amp_min)/(amp_max-amp_min) for s in amp_stims]
amp_conts = [(s-amp_min)/(amp_max-amp_min) for s in amp_conts]

pval = wilcoxon(amp_stims, amp_conts)[1]


fig, ax = plt.subplots(figsize=(3,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.set_xticklabels(['non-stim', 'stim'], minor=False)
ax.set_xticks([1,2])
ax.set_yticks([0,.5,1])

ax.set(ylabel='norm. theta amplitude')
fig.suptitle('theta amplitude, stim v stim-cont, p={}'.format(round(pval, 3)))

bp = ax.bar([1, 2], [np.mean(amp_starts), np.mean(amp_stims)],
            color=['grey', 'royalblue'], edgecolor=['k','k'], width=.35)
    
ax.scatter([[1]*len(amp_starts), [2]*len(amp_stims)], [amp_starts, amp_stims], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[1]*len(amp_starts), [2]*len(amp_stims)], [amp_starts, amp_stims], zorder=2,
        color='grey', alpha=.5)

ax.set(xlim=(0.5, 2.5), ylim=(-.01, 1.01))

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_theta_amp_bar.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)



#%% plot circular deviation stim v baseline
fig, ax = plt.subplots(figsize=(6,2))
fig.suptitle('avg. circ. dev., stim v baseline')

ax.set(xlabel='time (s)', ylabel='avg. circ. dev. (π)')
for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)

mean_stims_cir_dev = np.mean(all_stims_cir_dev, axis=0)
mean_start_cir_dev = np.mean(all_start_cir_dev, axis=0)
sem_stims_cir_dev = sem(all_stims_cir_dev, axis=0)
sem_start_cir_dev = sem(all_start_cir_dev, axis=0)

stimln, = ax.plot(taxis, mean_stims_cir_dev, 'royalblue')
baseln, = ax.plot(taxis, mean_start_cir_dev, 'grey', alpha=.5)
ax.fill_between(taxis, mean_stims_cir_dev+sem_stims_cir_dev,
                       mean_stims_cir_dev-sem_stims_cir_dev,
                       color='royalblue', alpha=.25)
ax.fill_between(taxis, mean_start_cir_dev+sem_start_cir_dev,
                       mean_start_cir_dev-sem_start_cir_dev,
                       color='grey', alpha=.25)

ax.legend([stimln, baseln], ['stim', 'baseline'], frameon=False)

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_baseline_theta_cir_dev.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


#%% plot circular deviation stim v stim cont
fig, ax = plt.subplots(figsize=(6,2.5))
fig.suptitle('avg. circ. dev., stim v shuffle')

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)

mean_stims_cir_dev = np.mean(all_stims_cir_dev, axis=0)
mean_shuf_cir_dev = np.mean(shuf_all_stims_cir_dev, axis=0)
sem_stims_cir_dev = sem(all_stims_cir_dev, axis=0)
sem_shuf_cir_dev = sem(shuf_all_stims_cir_dev, axis=0)

low = []; high = []
for sess in shuf_all_stims_cir_dev_perc:
    low.append(sess[0])
    high.append(sess[1])

shuf_low = np.mean(low, axis=0)
shuf_high = np.mean(high, axis=0)

stimln, = ax.plot(taxis, mean_stims_cir_dev, 'royalblue')
shufln, = ax.plot(taxis, mean_shuf_cir_dev, 'grey', alpha=.5)
ax.fill_between(taxis, mean_stims_cir_dev+sem_stims_cir_dev,
                       mean_stims_cir_dev-sem_stims_cir_dev,
                       color='royalblue', alpha=.25, edgecolor='none')
ax.fill_between(taxis, mean_shuf_cir_dev+sem_shuf_cir_dev,
                       mean_shuf_cir_dev-sem_shuf_cir_dev,
                       color='grey', alpha=.25, edgecolor='none')

# for i in range(6):
#     ax.fill_between([i*.083, i*.083+.083*0.3], [0,0], [1,1], color='royalblue', alpha=.25, edgecolor='none')
# ax.fill_between([0,.5], [0,0], [1,1], color='royalblue', alpha=.15, edgecolor='none')

ax.legend([stimln, shufln], ['stim', 'shuffle'], frameon=False)

ax.set(xlabel='time (s)', ylabel='avg. circ. dev. (π)',
       ylim=(.66,.92))

fig.tight_layout()
plt.show()

fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_shuf_theta_cir_dev.png',
            bbox_inches='tight',
            dpi=500)

plt.close(fig)


#%% theta frequencye stims v conts
fig, ax = plt.subplots(figsize=(3.5,3))

for p in ['top','right']:
    ax.spines[p].set_visible(False)
for p in ['left','bottom']:
    ax.spines[p].set_linewidth(1)
ax.set(ylabel='theta frequency', xlabel='time (s)',
       yticks=[0,.5,1])
fig.suptitle('theta frequency, stim v stim-cont')

mean_freq_stim = []
mean_freq_cont = []

for sess in all_stim_theta_frequency:
    mean_sess = np.mean(sess, axis=0)
    mean_freq_stim.append(mean_sess)
for sess in all_cont_theta_frequency:
    mean_sess = np.mean(sess, axis=0)
    mean_freq_cont.append(mean_sess)

all_mean_freq_stim = np.mean(mean_freq_stim, axis=0)
all_mean_freq_cont = np.mean(mean_freq_cont, axis=0)
mean_freq_max = max(max(all_mean_freq_stim), max(all_mean_freq_cont))
mean_freq_min = min(min(all_mean_freq_stim), min(all_mean_freq_cont))

# # min max normalisation
# mean_amp_stim = [(s-mean_amp_min)/(mean_amp_max-mean_amp_min) for s in mean_amp_stim]
# mean_amp_cont = [(s-mean_amp_min)/(mean_amp_max-mean_amp_min) for s in mean_amp_cont]

ax.plot(taxis, all_mean_freq_stim, 'royalblue')
ax.plot(taxis, all_mean_freq_cont, 'grey', alpha=.5)

fig.tight_layout()
plt.show()

# fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_theta_amp.png',
#             bbox_inches='tight',
#             dpi=500)

plt.close(fig)


#%%
freq_stim = []; freq_cont = []

for sess in mean_freq_stim:
    freq_stim.append(np.mean(sess[2500:3750]))  # from the start 
for sess in mean_freq_cont:
    freq_cont.append(np.mean(sess[2500:3750]))
freq_max = max(max(freq_stim), max(freq_cont))
freq_min = min(min(freq_stim), min(freq_cont))

# # min max normalisation
# amp_stims = [(s-amp_min)/(amp_max-amp_min) for s in amp_stims]
# amp_conts = [(s-amp_min)/(amp_max-amp_min) for s in amp_conts]

pval = wilcoxon(freq_stim, freq_cont)[1]


fig, ax = plt.subplots(figsize=(3,4))

for p in ['top', 'right', 'bottom']:
    ax.spines[p].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.set_xticklabels(['non-stim', 'stim'], minor=False)
ax.set_xticks([1,2])
ax.set_yticks([0,.5,1])

ax.set(ylabel='norm. theta amplitude')
fig.suptitle('theta amplitude, stim v stim-cont, p={}'.format(round(pval, 3)))

bp = ax.bar([1, 2], [np.mean(freq_cont), np.mean(freq_stim)],
            color=['grey', 'royalblue'], edgecolor=['k','k'], width=.35)
    
ax.scatter([[1]*len(freq_cont), [2]*len(freq_stim)], [freq_cont, freq_stim], zorder=2,
           s=15, color='grey', edgecolor='k', alpha=.5)
ax.plot([[1]*len(freq_cont), [2]*len(freq_stim)], [freq_cont, freq_stim], zorder=2,
        color='grey', alpha=.5)

ax.set(xlim=(0.5, 2.5))

fig.tight_layout()
plt.show()

# fig.savefig(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_theta_amp_bar.png',
#             bbox_inches='tight',
#             dpi=500)

plt.close(fig)