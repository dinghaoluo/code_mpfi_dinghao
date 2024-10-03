# -*- coding: utf-8 -*-
"""
Created on Wed 27 Sept 14:44:27 2023

compare all HPC cell's spiking profile between cont and stim 

Modified on Fri 10 Nov 

added ttest test to define responsive neurones

@author: Dinghao
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import sys 
import os 
import scipy.io as sio 
from scipy.stats import sem, ttest_rel

# plotting parameters 
xaxis = np.arange(-1250, 5000)/1250  # 1 s bef, 4 s after
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% suppress warnings...
# because we get this:
# UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
# which is super annoying 
import warnings
warnings.filterwarnings('ignore')


#%% load paths to recordings 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt


#%% dataframe to contain all results 
profiles = {'recname': [],
            'ctrl_pre_post': [],
            'stim_pre_post': [],
            'excited': [],
            'MI': [],
            'MI_early': [],
            'MI_late': [],
            # 'shuf_sig': [],
            # 'shuf_sig_early': [],
            # 'shuf_sig_late': [],
            'ctrl_mean': [],
            'ctrl_sem': [],
            'stim_mean': [],
            'stim_sem': []
                }

df = pd.DataFrame(profiles)


#%% list to contain shuffled MI's
shuf_MI = []
shuf_MIe = []
shuf_MIl = []


#%% main 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # load trains for this recording 
    all_info = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname),
                       allow_pickle=True).item()
    
    tot_time = 6 * 1250  # 6 seconds in 1250 Hz
    
    count_sensitive = 0  # how many cells in this session respond to stims
    count_exc = 0
    count_inh = 0
    
    trains = list(all_info.values())
    clu_list = list(all_info.keys())
    tot_trial = len(trains[0])
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    
    # depth 
    depth = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Depth.mat'.format(pathname, recname))['depthNeu'][0]
    rel_depth = depth['relDepthNeu'][0][0]
    
    # behaviour parameters 
    beh_info = info['beh'][0][0]
    behPar = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_behPar_msess1.mat'.format(pathname, recname))
    stimOn = behPar['behPar']['stimOn'][0][0][0][1:]
    stim_trials = np.where(stimOn!=0)[0]+1
    cont_trials = stim_trials+2
    tot_stim = len(stim_trials)
    
    # shuffle trial label
    pooled_trials = np.concatenate((stim_trials, cont_trials ))
    shuf_type1 = [];  shuf_type2 = []  # shuffled trial types 
    for i in range(100):  # shuffle 100 times
        shuf = np.random.permutation(pooled_trials)
        shuf_type1.append(shuf[:tot_stim])
        shuf_type2.append(shuf[tot_stim:])
    
    tot_clu = len(pyr_id)
    tot_pyr = sum(pyr_id)  # how many pyramidal cells are in this recording
    pyr_cont = {}
    pyr_stim = {}
    
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            d = rel_depth[i]
            temp_cont = np.zeros((len(cont_trials), tot_time))
            temp_stim = np.zeros((len(stim_trials), tot_time))
            for ind, trial in enumerate(cont_trials):
                trial_length = len(trains[i][trial])-1250
                if trial_length<tot_time and trial_length>0:
                    temp_cont[ind, :trial_length] = trains[i][trial][1250:8750]
                elif trial_length>0:
                    temp_cont[ind, :] = trains[i][trial][1250:8750]
            for ind, trial in enumerate(stim_trials):
                trial_length = len(trains[i][trial])-1250
                if trial_length<tot_time and trial_length>0:
                    temp_stim[ind, :trial_length] = trains[i][trial][1250:8750]
                elif trial_length>0:
                    temp_stim[ind, :] = trains[i][trial][1250:8750]
            
            # shuffled label, 15 June 2024, Dinghao 
            shuf_mod = []; shuf_mod_early = []; shuf_mod_late = []
            for shuf_i in range(100):  # shuffled labels
                type1 = []; type2= []
                type1e = []; type2e = []
                type1l = []; type2l = []
                for ind, trial in enumerate(shuf_type1[shuf_i]):
                    type1.append(np.mean(trains[i][trial][3125:4375], axis=0))
                    type1e.append(np.mean(trains[i][trial][2500:3750], axis=0))
                    type1l.append(np.mean(trains[i][trial][3750:5000], axis=0))
                for ind, trial in enumerate(shuf_type2[shuf_i]):
                    type2.append(np.mean(trains[i][trial][3125:4375], axis=0))
                    type2e.append(np.mean(trains[i][trial][2500:3750], axis=0))
                    type2l.append(np.mean(trains[i][trial][3750:5000], axis=0))
                    
                # single cell shuf storage
                shuf_mod.append((np.mean(type1)-np.mean(type2))/(np.mean(type1)+np.mean(type2)))
                shuf_mod_early.append((np.mean(type1e)-np.mean(type2e))/(np.mean(type1e)+np.mean(type2e)))
                shuf_mod_late.append((np.mean(type1l)-np.mean(type2l))/(np.mean(type1e)+np.mean(type2l)))
                
                # store them into the overall shuf containers 
                shuf_MI.append((np.mean(type1)-np.mean(type2))/(np.mean(type1)+np.mean(type2)))
                shuf_MIe.append((np.mean(type1e)-np.mean(type2e))/(np.mean(type1e)+np.mean(type2e)))
                shuf_MIl.append((np.mean(type1l)-np.mean(type2l))/(np.mean(type1e)+np.mean(type2l)))
                
            # sig_threshold = np.percentile(shuf_mod, [5, 95])  # 95%
            # sig_threshold_early = np.percentile(shuf_mod_early, [5, 95])
            # sig_threshold_late = np.percentile(shuf_mod_late, [5, 95])
            
            # plotting 
            fig, axs = plt.subplot_mosaic('AAB', figsize=(4,2.8))
            
            # mean profile 
            mean_prof_cont = np.mean(temp_cont, axis=0)*1250
            sem_prof_cont = sem(temp_cont, axis=0)*1250
            mean_prof_stim = np.mean(temp_stim, axis=0)*1250
            sem_prof_stim = sem(temp_stim, axis=0)*1250
            
            # pre-post ratio, 19 June 2024, Dinghao  
            cont_pp_ratio = np.nanmean(mean_prof_cont[625:1875])/np.nanmean(mean_prof_cont[3125:4375])  # .5~1.5s
            stim_pp_ratio = np.nanmean(mean_prof_stim[625:1875])/np.nanmean(mean_prof_stim[3125:4375])  # .5~1.5s
            
            contln, = axs['A'].plot(xaxis, mean_prof_cont[1250:], color='grey')
            axs['A'].fill_between(xaxis, mean_prof_cont[1250:]+sem_prof_cont[1250:],
                                         mean_prof_cont[1250:]-sem_prof_cont[1250:],
                                  alpha=.25, color='grey', edgecolor='none')
            stimln, = axs['A'].plot(xaxis, mean_prof_stim[1250:], color='royalblue')
            axs['A'].fill_between(xaxis, mean_prof_stim[1250:]+sem_prof_stim[1250:],
                                         mean_prof_stim[1250:]-sem_prof_stim[1250:],
                                  alpha=.25, color='royalblue', edgecolor='none')
            
            axs['A'].legend([contln, stimln], ['ctrl', 'stim'], 
                            frameon=False, fontsize=8)
            for p in ['top', 'right']:
                axs['A'].spines[p].set_visible(False)
            
            fig.suptitle(cluname)
            axs['A'].set(title='depth={}'.format(d),
                         xlabel='time (s)', ylabel='spike rate (Hz)',
                         xlim=(-1, 4))
            
            # sensitivity 
            # 27 Mar 24, tests for each time bin
            test_window = 2 * 1250  # 2 s
            cont_rate = temp_cont[:,2500:2500+test_window]
            stim_rate = temp_stim[:,2500:2500+test_window]
            bar_y = max(max(mean_prof_cont), max(mean_prof_stim)) * 1.1
            
            sig_bins = []
            for i in range(test_window):
                p_curr = ttest_rel(a=cont_rate[:,i], b=stim_rate[:,i])[1]
                if p_curr<0.05:
                    sig_bins.append(i/1250)
            # axs['A'].scatter(sig_bins, [bar_y]*len(sig_bins), s=3, c='r')
            
            cont_rate_mean = np.mean(temp_cont[:, 3125:4375], axis=1)*1250  # 0.5-1.5 s mean 
            stim_rate_mean = np.mean(temp_stim[:, 3125:4375], axis=1)*1250
            cont_rate_mean_early = np.mean(temp_cont[:, 2500:3750], axis=1)*1250  # 0-1 s mean
            stim_rate_mean_early = np.mean(temp_stim[:, 2500:3750], axis=1)*1250
            cont_rate_mean_late = np.mean(temp_cont[:, 3750:5000], axis=1)*1250  # 1-2 s mean 
            stim_rate_mean_late = np.mean(temp_stim[:, 3750:5000], axis=1)*1250
            
            # modulation index
            stim_mom = np.mean(stim_rate_mean)
            cont_mom = np.mean(cont_rate_mean)
            mod = (stim_mom-cont_mom)/(stim_mom+cont_mom)
            # if mod<=sig_threshold[0] or mod >=sig_threshold[1]:
            #     shuf_sig = True
            # else:
            #     shuf_sig = False
            
            stim_mom_early = np.mean(stim_rate_mean_early)
            cont_mom_early = np.mean(cont_rate_mean_early)
            mod_early = (stim_mom_early-cont_mom_early)/(stim_mom_early+cont_mom_early)
            # if mod_early<=sig_threshold_early[0] or mod_early>=sig_threshold_early[1]:
            #     shuf_sig_early = True
            # else:
            #     shuf_sig_early = False
            
            stim_mom_late = np.mean(stim_rate_mean_late)
            cont_mom_late = np.mean(cont_rate_mean_late)
            mod_late = (stim_mom_late-cont_mom_late)/(stim_mom_late+cont_mom_late)
            # if mod_late<=sig_threshold_late[0] or mod_late>=sig_threshold_late[1]:
            #     shuf_sig_late = True
            # else:
            #     shuf_sig_late = False
            
            t_res = ttest_rel(a=cont_rate_mean, b=stim_rate_mean)
            pval = t_res[1]
                
            for p in ['top', 'right', 'bottom']:
                axs['B'].spines[p].set_visible(False)
            axs['B'].set_xticklabels(['stim-\ncont', 'stim'], minor=False)
            axs['B'].set(ylabel='spike rate (Hz)',
                         title='stim-cont v stim\np={}'.format(round(pval, 4)))

            bp = axs['B'].boxplot([cont_rate_mean, stim_rate_mean],
                                  positions=[.5, 1],
                                  patch_artist=True,
                                  notch='True')
            colors = ['grey', 'royalblue']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            bp['fliers'][0].set(marker ='v',
                            color ='#e7298a',
                            markersize=2,
                            alpha=0.5)
            bp['fliers'][1].set(marker ='o',
                            color ='#e7298a',
                            markersize=2,
                            alpha=0.5)
            for median in bp['medians']:
                median.set(color='darkblue',
                           linewidth=1)
            
            fig.tight_layout()
            plt.show()
            
            # make folders if not exist
            outdirroot = 'Z:\Dinghao\code_dinghao\HPC_all\{}'.format(recname)
            if not os.path.exists(outdirroot):
                os.makedirs(outdirroot)
            outdir = '{}\stim_stimcont_pyr_only'.format(outdirroot)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            fig.savefig('{}\{}'.format(outdir, cluname),
                        dpi=500,
                        bbox_inches='tight')
            fig.savefig('{}\{}.pdf'.format(outdir, cluname),
                        bbox_inches='tight')
             
            plt.close(fig)
            
            # 27 Mar 24, put into dataframe
            df.loc[cluname] = np.asarray([recname, 
                                          cont_pp_ratio,
                                          stim_pp_ratio,
                                          np.mean(cont_rate_mean)<np.mean(stim_rate_mean), 
                                          mod, 
                                          mod_early, 
                                          mod_late,
                                          # shuf_sig,
                                          # shuf_sig_early,
                                          # shuf_sig_late,
                                          mean_prof_cont,
                                          sem_prof_cont,
                                          mean_prof_stim,
                                          sem_prof_stim
                                          ],
                                         dtype='object')

    
#%% save dataframe 
df.to_csv(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv')
df.to_pickle(r'Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl')


#%% save shuffled MI distributions after converting to numpy arrays and filtering out nan's
shuf_MI=np.array(shuf_MI); shuf_MIe=np.array(shuf_MIe); shuf_MIl=np.array(shuf_MIl)
shuf_MI=shuf_MI[~np.isnan(shuf_MI)]; shuf_MIe=shuf_MIe[~np.isnan(shuf_MIe)]; shuf_MIl=shuf_MIl[~np.isnan(shuf_MIl)]
np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_shuf_MI_pyr_only.npy', shuf_MI)
np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_shuf_MIe_pyr_only.npy', shuf_MIe)
np.save('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_shuf_MIl_pyr_only.npy', shuf_MIl)