# -*- coding: utf-8 -*-
"""
Created on Thur 11 July 17:26:45 2024

plot and classify each LC cell as significantly responding to licks/rewards 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import pandas as pd
from scipy.stats import ttest_rel, ranksums, wilcoxon

# plotting parameters
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load data 
rasters = np.load('Z:\Dinghao\code_dinghao\LC_all\LC_all_rasters_simp_name.npy',
                  allow_pickle=True).item()
all_train = np.load('Z:/Dinghao/code_dinghao/LC_all/LC_all_info.npy',
                    allow_pickle=True).item()

cell_prop = pd.read_pickle('Z:\Dinghao\code_dinghao\LC_all\LC_all_single_cell_properties.pkl')


#%% specify RO peaking putative Dbh cells
clu_list = list(cell_prop.index)

tag_list = []; put_list = []
tag_rop_list = []; put_rop_list = []
for clu in cell_prop.index:
    tg = cell_prop['tagged'][clu]
    pt = cell_prop['putative'][clu]
    rop = cell_prop['peakness'][clu]
    
    if tg:
        tag_list.append(clu)
        if rop:
            tag_rop_list.append(clu)
    if pt:
        put_list.append(clu)
        if rop:
            put_rop_list.append(clu)


#%% shuffle function 
def cir_shuf(train, length, num_shuf=1000):
    tot_t = len(train)
    shuf_array = np.zeros([num_shuf, length])
    for i in range(num_shuf):
        rand_shift = np.random.randint(1, tot_t/2)
        shuf_array[i,:][:tot_t] = np.roll(train, -rand_shift)[:length]
    
    return np.mean(shuf_array, axis=0)


#%% gaussian filter 
gx_spike = np.arange(-500, 500, 1)
sigma_spike = 1250/3
gaus_spike = [1 / (sigma_spike*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_spike**2)) for x in gx_spike]


#%% container 
profiles = {'cluname': [],
            'tagged': [],
            'putative': [],
            'lick_activated': [],
            'lick_ttest': [],
            'lick_ratio_ranksums': [],
            'lick_tp_std': [],
            'rew_activated': [],
            'rew_ttest': [],
            'rew_ratio_ranksums': [],
            'rew_tp_std': []
                }

df = pd.DataFrame(profiles)


#%% MAIN 
for cluname in clu_list:
    print(cluname)
    raster = rasters[cluname]
    train = all_train[cluname]
    
    filename = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_alignRun_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    alignRun = sio.loadmat(filename)
    
    licks = alignRun['trialsRun']['lickLfpInd'][0][0][0][1:]
    starts = alignRun['trialsRun']['startLfpInd'][0][0][0][1:]
    pumps = alignRun['trialsRun']['pumpLfpInd'][0][0][0][1:]
    tot_trial = licks.shape[0]
    for trial in range(tot_trial):
        if len(pumps[trial])>0:
            pumps[trial] = pumps[trial][0] - starts[trial]
        else:
            pumps[trial] = [10000]
    
    behParf = 'Z:/Dinghao/MiceExp/ANMD{}/{}/{}/{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(cluname[1:5], cluname[:14], cluname[:17], cluname[:17])
    behPar = sio.loadmat(behParf)
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stimOn_ind = np.where(stimOn!=0)[0]+1
    
    first_licks = []
    for trial in range(tot_trial):
        lk = [l for l in licks[trial] if l-starts[trial] > 1250]  # only if the animal does not lick in the first second (carry-over licks)
        if len(lk)==0:
            first_licks.append(10000)
        else:
            first_licks.extend(lk[0]-starts[trial])

    # plotting
    fig, axs = plt.subplot_mosaic('AAAA;AAAA;AAAA;BCDE',figsize=(5,5))
    axs['A'].set(xlabel='time (s)', ylabel='trial # by first licks',
                 xlim=(-1, 7))
    for p in ['top', 'right']:
        axs['A'].spines[p].set_visible(False)

    # containers 
    lick_pre_rate = []; lick_post_rate = []
    lick_pre_rate_shuf = []; lick_post_rate_shuf = []
    lick_ratio = []; lick_ratio_shuf = []
    rew_pre_rate = []; rew_post_rate = []
    rew_pre_rate_shuf = []; rew_post_rate_shuf = []
    rew_ratio = []; rew_ratio_shuf = []
    
    lick_t_points = []  # transition points from activity to non-activity (vice versa)
    rew_t_points = []
    
    around_lick = []
    around_rew = []
    
    for trial in range(tot_trial):
        curr_raster = raster[trial]
        curr_train = train[trial]
        
        # shuffle train for later use 
        length = len(curr_train)
        shuf_train = cir_shuf(curr_train, length, num_shuf=100)
        
        
        # lick sensitivity
        lick_window = [first_licks[trial]+3750-625, first_licks[trial]+3750, first_licks[trial]+3750+625]
        lick_pre_rate.append(sum(curr_train[lick_window[0]:lick_window[1]])*2)  # times 2 because it's half a second
        lick_post_rate.append(sum(curr_train[lick_window[1]:lick_window[2]])*2)
        lick_pre_rate_shuf.append(sum(shuf_train[lick_window[0]:lick_window[1]])*2)
        lick_post_rate_shuf.append(sum(shuf_train[lick_window[1]:lick_window[2]])*2)   
        if sum(curr_train[lick_window[1]:lick_window[2]])*2!=0 and sum(curr_train[lick_window[0]:lick_window[1]])*2!=0 and sum(shuf_train[lick_window[1]:lick_window[2]])*2!=0 and sum(shuf_train[lick_window[0]:lick_window[1]])*2!=0:
            lick_ratio.append(sum(curr_train[lick_window[0]:lick_window[1]])*2/sum(curr_train[lick_window[1]:lick_window[2]])*2)
            lick_ratio_shuf.append(sum(shuf_train[lick_window[0]:lick_window[1]])*2/sum(shuf_train[lick_window[1]:lick_window[2]])*2)         
        
        # reward sensitivity
        rew_window = [pumps[trial][0]+3750-625, pumps[trial][0]+3750, pumps[trial][0]+3750+625]
        rew_pre_rate.append(sum(curr_train[rew_window[0]:rew_window[1]])*2)  # times 2 because it's half a second
        rew_post_rate.append(sum(curr_train[rew_window[1]:rew_window[2]])*2)
        rew_pre_rate_shuf.append(sum(shuf_train[rew_window[0]:rew_window[1]])*2)
        rew_post_rate_shuf.append(sum(shuf_train[rew_window[1]:rew_window[2]])*2)
        if sum(curr_train[rew_window[1]:rew_window[2]])*2!=0 and sum(curr_train[rew_window[0]:rew_window[1]])*2!=0 and sum(shuf_train[rew_window[1]:rew_window[2]])*2!=0 and sum(shuf_train[rew_window[0]:rew_window[1]])*2!=0:
            rew_ratio.append(sum(curr_train[rew_window[0]:rew_window[1]])*2/sum(curr_train[rew_window[1]:rew_window[2]])*2)
            rew_ratio_shuf.append(sum(shuf_train[rew_window[0]:rew_window[1]])*2/sum(shuf_train[rew_window[1]:rew_window[2]])*2)
        
        # calculate transition points 
        if first_licks[trial]<10000:  # only execute if there is actually licks in this trial
            smooth_train = np.convolve(curr_train, gaus_spike, 'same')[first_licks[trial]+3750-625:first_licks[trial]+3750+625]
            around_lick.append(smooth_train)
            smooth_shuf_train = cir_shuf(smooth_train, 1250, num_shuf=100)
            # idea: get 1 mean shuffled value and use next() to find the crossing point
            mean_shuf = np.mean(smooth_shuf_train)
            try:
                if np.mean(smooth_train[first_licks[trial]+3750-625:first_licks[trial]+3750])<np.mean(smooth_train[first_licks[trial]+3750:first_licks[trial]+3750+625]):  # excited
                    tp = next(k for k, value in enumerate(smooth_train) if value < mean_shuf) + first_licks[trial]
                    lick_t_points.append(tp-first_licks[trial])
                else:  # inhibited 
                    tp = next(k for k, value in enumerate(smooth_train) if value < mean_shuf) + first_licks[trial]
                    lick_t_points.append(tp-first_licks[trial])
            except:
                lick_t_points.append(np.nan)  # if StopIteration, default to nan to eliminate later
        else:  # if no licks in the current trial, default to nan
            lick_t_points.append(np.nan)
        lick_tp_std = np.nanstd(lick_t_points)
            
        # calculate transition points (rew)
        if pumps[trial][0]<10000:  # only execute if there is actually licks in this trial
            smooth_train = np.convolve(curr_train, gaus_spike, 'same')[pumps[trial][0]+3750-625:pumps[trial][0]+3750+625]
            around_rew.append(smooth_train)
            smooth_shuf_train = cir_shuf(smooth_train, 1250, num_shuf=100)
            mean_shuf = np.mean(smooth_shuf_train)
            try:
                if np.mean(smooth_train[pumps[trial][0]+3750-625:pumps[trial][0]+3750])<np.mean(smooth_train[pumps[trial][0]+3750:pumps[trial][0]+3750+625]):  # excited
                    tp = next(k for k, value in enumerate(smooth_train) if value > mean_shuf) + pumps[trial][0]
                    rew_t_points.append(tp-pumps[trial][0])
                else:  # inhibited 
                    tp = next(k for k, value in enumerate(smooth_train) if value < mean_shuf) + pumps[trial][0]
                    rew_t_points.append(tp-pumps[trial][0])
            except:
                rew_t_points.append(np.nan)
        else:
            rew_t_points.append(np.nan)
        rew_tp_std = np.nanstd(rew_t_points)
        
        # prepare rasters
        curr_trial = np.where(raster[trial]==1)[0]
        curr_trial = [(s-3750)/1250 for s in curr_trial if s>2500]  # starts from -1 s 
        
        c = 'grey'
        calpha = 0.7
        dotsize = 0.35
        if stimOn[trial]==1:
            c = 'royalblue'
            calpha = 1.0
            dotsize = 0.55
        
        axs['A'].scatter(curr_trial, [trial+1]*len(curr_trial),
                         color=c, alpha=calpha, s=dotsize)
        axs['A'].plot([first_licks[trial]/1250, first_licks[trial]/1250],
                      [trial, trial+1],
                      linewidth=2, color='orchid')
        axs['A'].plot([pumps[trial][0]/1250, pumps[trial][0]/1250],
                      [trial, trial+1],
                      linewidth=2, color='darkgreen')
     
    fl, = axs['A'].plot([],[],color='orchid',label='1st licks')
    pp, = axs['A'].plot([],[],color='darkgreen',alpha=.35,label='rew.')
    axs['A'].legend(handles=[fl, pp], frameon=False, fontsize=8)
    axs['A'].set(xticks=[0, 2, 4],
                 xlim=(-1, 6))
    
    # lick activated/inhibited 
    around_lick = [train for train in around_lick if len(train)==1250]  # filter to eliminate trials that are too short 
    lick_mean_train = np.mean(around_lick, axis=0)
    if np.mean(lick_mean_train[:625])>np.mean(lick_mean_train[625:]):
        lick_activated = False
    else:
        lick_activated = True
    around_rew = [train for train in around_rew if len(train)==1250]  # same as above 
    rew_mean_train = np.mean(around_rew, axis=0)
    if np.mean(rew_mean_train[:625])>np.mean(rew_mean_train[625:]):
        rew_activated = False
    else:
        rew_activated = True
    
    
    # statistics 
    lick_rate_ttest_pval = ttest_rel(lick_pre_rate, lick_post_rate)[1]
    lick_ratio_ranksums_pval = ranksums(lick_ratio, lick_ratio_shuf)[1]
    rew_rate_ttest_pval = ttest_rel(rew_pre_rate, rew_post_rate)[1]
    rew_ratio_ranksums_pval = ranksums(rew_ratio, rew_ratio_shuf)[1]

    for p in ['top', 'right', 'bottom']:
        for a in ['B','C','D','E']:
            axs[a].spines[p].set_visible(False)
    for a in ['B','D']:
        axs[a].set(ylabel='spike rate (Hz)')
    for a in ['C','E']:
        axs[a].set(ylabel='pre-post ratio')
        
    # plot B
    axs['B'].set(xticklabels=(['pre-\n1st-lick', 'post-\n1st-lick']),
                 title='ttest\np={}'.format(round(lick_rate_ttest_pval, 5)))
    vpb = axs['B'].violinplot([lick_pre_rate, lick_post_rate],
                              positions=[1, 2],
                              showextrema=False)
    vpb['bodies'][0].set_color('thistle')
    vpb['bodies'][1].set_color('plum')
    for i in [0,1]:
        vpb['bodies'][i].set_edgecolor('none')
        vpb['bodies'][i].set_alpha(.75)
        b = vpb['bodies'][i]
        m = np.mean(b.get_paths()[0].vertices[:,0])
        if i==0:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
        if i==1:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
    axs['B'].scatter([1.1]*len(lick_pre_rate), 
                     lick_pre_rate, 
                     s=10, c='thistle', ec='none', lw=.5, alpha=.2)
    axs['B'].scatter([1.9]*len(lick_post_rate), 
                     lick_post_rate, 
                     s=10, c='plum', ec='none', lw=.5, alpha=.2)
    axs['B'].plot([[1.1]*len(lick_pre_rate), [1.9]*len(lick_post_rate)], [lick_pre_rate, lick_post_rate], 
                  color='grey', alpha=.2, linewidth=1)
    axs['B'].plot([1.1, 1.9], [np.median(lick_pre_rate), np.median(lick_post_rate)],
                  color='grey', linewidth=2)
    axs['B'].scatter(1.1, np.median(lick_pre_rate), 
                     s=30, c='thistle', ec='none', lw=.5, zorder=2)
    axs['B'].scatter(1.9, np.median(lick_post_rate), 
                     s=30, c='plum', ec='none', lw=.5, zorder=2)

    # plot C
    axs['C'].set(xticklabels=['pre-\npost', 'pre-\npost-\nshuf'],
                 title='ranksums\np={}'.format(round(lick_ratio_ranksums_pval, 5)))
    vpc = axs['C'].violinplot([lick_ratio, lick_ratio_shuf],
                              positions=[1, 2],
                              showextrema=False)
    vpc['bodies'][0].set_color('grey')
    vpc['bodies'][1].set_color('gainsboro')
    for i in [0,1]:
        vpc['bodies'][i].set_edgecolor('none')
        vpc['bodies'][i].set_alpha(.75)
        b = vpc['bodies'][i]
        m = np.mean(b.get_paths()[0].vertices[:,0])
        if i==0:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
        if i==1:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
    axs['C'].scatter([1.1]*len(lick_ratio), 
                     lick_ratio, 
                     s=10, c='grey', ec='none', lw=.5, alpha=.2)
    axs['C'].scatter([1.9]*len(lick_ratio_shuf), 
                     lick_ratio_shuf, 
                     s=10, c='gainsboro', ec='none', lw=.5, alpha=.2)
    axs['C'].plot([[1.1]*len(lick_ratio), [1.9]*len(lick_ratio_shuf)], [lick_ratio, lick_ratio_shuf], 
                  color='grey', alpha=.2, linewidth=1)
    axs['C'].plot([1.1, 1.9], [np.median(lick_ratio), np.median(lick_ratio_shuf)],
                  color='grey', linewidth=2)
    axs['C'].scatter(1.1, np.median(lick_ratio), 
                     s=30, c='grey', ec='none', lw=.5, zorder=2)
    axs['C'].scatter(1.9, np.median(lick_ratio_shuf), 
                     s=30, c='gainsboro', ec='none', lw=.5, zorder=2)
    
    # plot D
    axs['D'].set(xticklabels=(['pre-\nrew.', 'post-\nrew.']),
                 title='ttest\np={}'.format(round(rew_rate_ttest_pval, 5)))
    vpd = axs['D'].violinplot([rew_pre_rate, rew_post_rate],
                              positions=[1, 2],
                              showextrema=False)
    vpd['bodies'][0].set_color('yellowgreen')
    vpd['bodies'][1].set_color('forestgreen')
    for i in [0,1]:
        vpd['bodies'][i].set_edgecolor('none')
        vpd['bodies'][i].set_alpha(.75)
        b = vpd['bodies'][i]
        m = np.mean(b.get_paths()[0].vertices[:,0])
        if i==0:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
        if i==1:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
    axs['D'].scatter([1.1]*len(rew_pre_rate), 
                     rew_pre_rate, 
                     s=10, c='yellowgreen', ec='none', lw=.5, alpha=.2)
    axs['D'].scatter([1.9]*len(rew_post_rate), 
                     rew_post_rate, 
                     s=10, c='forestgreen', ec='none', lw=.5, alpha=.2)
    axs['D'].plot([[1.1]*len(rew_pre_rate), [1.9]*len(rew_post_rate)], [rew_pre_rate, rew_post_rate], 
                  color='grey', alpha=.2, linewidth=1)
    axs['D'].plot([1.1, 1.9], [np.median(rew_pre_rate), np.median(rew_post_rate)],
                  color='grey', linewidth=2)
    axs['D'].scatter(1.1, np.median(rew_pre_rate), 
                     s=30, c='yellowgreen', ec='none', lw=.5, zorder=2)
    axs['D'].scatter(1.9, np.median(rew_post_rate), 
                     s=30, c='forestgreen', ec='none', lw=.5, zorder=2)

    # plot E
    axs['E'].set(xticklabels=['pre-\npost', 'pre-\npost-\nshuf'],
                 title='ranksums\np={}'.format(round(rew_ratio_ranksums_pval, 5)))
    vpc = axs['E'].violinplot([rew_ratio, rew_ratio_shuf],
                              positions=[1, 2],
                              showextrema=False)
    vpc['bodies'][0].set_color('grey')
    vpc['bodies'][1].set_color('gainsboro')
    for i in [0,1]:
        vpc['bodies'][i].set_edgecolor('none')
        vpc['bodies'][i].set_alpha(.75)
        b = vpc['bodies'][i]
        m = np.mean(b.get_paths()[0].vertices[:,0])
        if i==0:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
        if i==1:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)
    axs['E'].scatter([1.1]*len(rew_ratio), 
                     rew_ratio, 
                     s=10, c='grey', ec='none', lw=.5, alpha=.2)
    axs['E'].scatter([1.9]*len(rew_ratio_shuf), 
                     rew_ratio_shuf, 
                     s=10, c='gainsboro', ec='none', lw=.5, alpha=.2)
    axs['E'].plot([[1.1]*len(rew_ratio), [1.9]*len(rew_ratio_shuf)], [rew_ratio, rew_ratio_shuf], 
                  color='grey', alpha=.2, linewidth=1)
    axs['E'].plot([1.1, 1.9], [np.median(rew_ratio), np.median(rew_ratio_shuf)],
                  color='grey', linewidth=2)
    axs['E'].scatter(1.1, np.median(rew_ratio), 
                     s=30, c='grey', ec='none', lw=.5, zorder=2)
    axs['E'].scatter(1.9, np.median(rew_ratio_shuf), 
                     s=30, c='gainsboro', ec='none', lw=.5, zorder=2)
    
    clutag = ''
    if cluname in tag_list: clutag = 'tagged'
    if cluname in put_list: clutag = 'putative'
    
    fig.suptitle('{} {}\nlick_tp_std={}\nrew_tp_std={}'.format(cluname, clutag, lick_tp_std, rew_tp_std))
    fig.tight_layout()
    plt.show()
    
    if cluname in tag_list:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_with_sensitivity\{}_tagged.png'.format(cluname),
                    dpi=300)
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_with_sensitivity\{}_tagged.pdf'.format(cluname))
    elif cluname in put_list:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_with_sensitivity\{}_putative.png'.format(cluname),
                    dpi=300)
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_with_sensitivity\{}_putative.pdf'.format(cluname))
    else:
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_with_sensitivity\{}.png'.format(cluname),
                    dpi=300)
        fig.savefig('Z:\Dinghao\code_dinghao\LC_all\single_cell_raster_with_sensitivity\{}.pdf'.format(cluname))
    
    plt.close(fig)

    df.loc[cluname] = np.asarray([cluname, 
                                  cluname in tag_list,
                                  cluname in put_list,
                                  lick_activated,
                                  lick_rate_ttest_pval,
                                  lick_ratio_ranksums_pval,
                                  lick_tp_std,
                                  rew_activated,
                                  rew_rate_ttest_pval,
                                  rew_ratio_ranksums_pval,
                                  rew_tp_std
                                  ],
                                 dtype='object')
    

#%% classfication 
for cluname in clu_list:
    curr_properties = df.loc[cluname]
    lick_activated = curr_properties['lick_activated']
    rew_activated = curr_properties['rew_activated']
    