# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:07:06 2024
Modified on Tue Apr 2 14:17:12 2024

@author: Jingyu Cao; modified by Dinghao Luo 
"""


#%% imports 
import os
import numpy as np
import scipy 
import mat73
import sys
import matplotlib.pyplot as plt
import math

if ('Z:/Dinghao/code_dinghao/common' in sys.path) == False:
    sys.path.append('Z:/Dinghao/code_dinghao/common')
from common import normalise



#%% define functions 
def moving_average(arr, window_size=3):
    
    frame = 0
    moving_averages = []
    while frame < len(arr)-window_size+1:
        window_average = np.sum(arr[frame:frame+window_size]) / window_size
        moving_averages.append(window_average)
        frame+=1
        
    return moving_averages


def load(path):
    mat = mat73.loadmat(path)
    
    mat_real = mat['trials']
    cluList = mat['cluList']['localClu']
    delay = mat_real[1]['trackLen']
    
    # extract dFF
    dFF = []
    for i in range (0,len(mat_real)):
        dFF.append(mat_real[i]['dFF'])   
    
    #extract dFFgf
    dFFgf = []
    for i in range (0,len(mat_real)):
        dFFgf.append(mat_real[i]['dFFgf'])
    
    #extract dFFSM
    dFFSM = []
    for i in range (0,len(mat_real)):
        dFFSM.append(mat_real[i]['dFFSM'])
    
    #extract Fc
    # Fc=[]
    # for i in range (0,len(mat_real)):
    #     Fc.append(mat_real[i]['F'])
    
    #extract behaviour activity
    licks = []
    for i in range (0,len(mat_real)):
        licks.append(mat_real[i]['lickLfpInd'])
    
    Rews = []
    for i in range (0,len(mat_real)):
        Rews.append(mat_real[i]['pumpLfpInd'])
    
    starts = []
    for i in range (0,len(mat_real)):
        starts.append(mat_real[i]['lfpIndStart'])
      
    ends = []
    for i in range (0,len(mat_real)):
        ends.append(mat_real[i]['lfpIndEnd'])

    return {'cluList':cluList, 'dFFgf':dFFgf, 'dFF':dFF,'dFFSM':dFFSM},  {'delay': delay, 'licks':licks, 'Rew':Rews, 'starts':starts, 'ends':ends}
    # return {'dFFgf':dFFgf, 'dFFSM':dFFSM},  {'delay': delay, 'licks':licks, 'Rew':Rews, 'starts':starts, 'ends':ends}

def get_roi(chan1, ROIid):
    
    return chan1['cluList'].astype(int).tolist().index(ROIid+1)

def get_id(chan1, ROI_seq):
    
    return int(chan1['cluList'][ROI_seq])-1

def Ave_trial(chan, data, sem, quality, min_length, tot_ROI, tot_trial): #get trial_ave for each roi with SEM 
    
    # tot_ROI = chan[data][1].shape[1]
    # tot_trial = len(chan[data])
    # min_length = min((chan[data][t][:,0]).shape[0] for t in range(tot_trial))
    
    list_ave_ROIs = []
    list_sem_ROIs = []

    for r in range(tot_ROI):
        
        roi_list_trials = np.empty([min_length, tot_trial])
      
        for t in range(tot_trial):
            roi_list_trials[:,t] = chan[data][t][:min_length,r]
        
        list_ave = np.mean(roi_list_trials, axis = 1)
        list_ave_ROIs.append(list_ave)
        
        chan.update({data+'_ave': list_ave_ROIs})
        
        if sem == 1:
            list_sem = scipy.stats.sem(roi_list_trials, axis = 1)
            list_sem_ROIs.append(list_sem)
            
            chan.update({data+'_sem': list_sem_ROIs})

    if quality ==1:
        
        list1_ave_ROIs = []
        list1_sem_ROIs = []
        list2_ave_ROIs = []
        list2_sem_ROIs = []
          
        for r in range(tot_ROI):
            
            roi_list1_trials = np.empty([min_length, len(good_trials)])
            roi_list2_trials = np.empty([min_length, len(bad_trials)])
            a=0
            b=0
            for t in good_trials:
                
                roi_list1_trials[:,a] = chan[data][t][:min_length,r]
                a += 1
                
            
            list1_ave = np.mean(roi_list1_trials, axis = 1)
            list1_ave_ROIs.append(list1_ave)
            
            for t in bad_trials:
                
                roi_list2_trials[:,b] = chan[data][t][:min_length,r]
                b += 1
            
            list2_ave = np.mean(roi_list2_trials, axis = 1)
            list2_ave_ROIs.append(list2_ave)
            
            chan.update({data+'_ave'+'_good': list1_ave_ROIs, data+'_ave'+'_bad': list2_ave_ROIs})
            
            if sem == 1:
                list1_sem = scipy.stats.sem(roi_list1_trials, axis = 1)
                list1_sem_ROIs.append(list1_sem)
                
                list2_sem = scipy.stats.sem(roi_list2_trials, axis = 1)
                list2_sem_ROIs.append(list2_sem)
                
                chan.update({data+'_sem_good': list1_sem_ROIs, data+'_sem_bad': list2_sem_ROIs})

def Plot_ave(data, shuff, save):
    plots = 50 #plots per figure
    fig_num = math.ceil(tot_ROI/plots)
    
    delay = Beh['delay']*10**-6
    
    for i in range(fig_num):
        roi_col = 5
        roi_row = math.ceil(plots/roi_col)
        
        min_length = len(chan1[data+'_ave'][0])
        fig = plt.figure(i, figsize=[roi_col*12, roi_row*5]); fig.tight_layout()
        #fig.subplots_adjust(top=1.5)
        fig.suptitle(f'All ROIs_TrialAve__Runave{run_win}_{Channel}-{Filebase}_Shuff={shuff}', size=15)
        fig.subplots_adjust(top=0.97)
    
    
        for r in range(plots*i, min(plots*(i+1), tot_ROI)):
            xaxis = np.arange(min_length) / 500
            plot_pos = np.arange(1, min(plots*(i+1),tot_ROI)+1)
            ax1 = fig.add_subplot(roi_row, roi_col, plot_pos[r-plots*i])
            subtitle = 'ROI' + str(int(chan1['cluList'][r])-1) + '(' + str(r) + ')'
            
            chan1_mean = chan1[data+'_ave'][r]
            chan2_mean = chan2[data+'_ave'][r]
            
            chan1_sem = chan1[data+'_sem'][r]
            chan2_sem = chan2[data+'_sem'][r]
            
            
            y_max1 = max(0.1,max(chan1_mean+chan1_sem))+0.01
            y_min1 = min(chan1_mean-chan1_sem)-0.01
            
            # y_max2 = max(chan2_mean)+0.1
            # y_min2 = min(chan2_mean)-0.1
            
            ax1.title.set_text(subtitle)
            ax1.set(ylabel= 'chan1'+data, xlabel='time (s)',
                    xlim=(0, min_length/500), ylim=(y_min1, y_max1))
            ax1.plot(xaxis, chan1_mean, color='green')
            ax1.plot(xaxis, chan2_mean, color="#800080")
            
            ax1.fill_between(xaxis,chan1_mean-chan1_sem, chan1_mean+chan1_sem, color='green', alpha = 0.1)
            #ax1.get_yaxis().set_visible(False)
            ax1.fill_between(xaxis,chan2_mean-chan2_sem, chan2_mean+chan2_sem, color="#800080", alpha = 0.1 )
            #ax4.get_yaxis().set_visible(False)
            
            #plot cue and reward zone
            ax1.axvspan(1+delay,2+delay, color = 'orange', alpha=0.1, label="Reward Zone")
            ax1.axvspan(0, 1, color = 'grey', alpha=0.1, label = 'Cue')
            
            
            if shuff ==1:
            
                chan1_shuff_mean = chan1[data+'_shuff'][:,r]
                chan1_shuff_95 = chan1[data+'_shuff'+'_95'][:,r]
                chan1_shuff_5 = chan1[data+'_shuff'+'_5'][:,r]
                #plot shuffled data mean
                ax1.plot(xaxis, chan1_shuff_mean, color="grey", linewidth=0.5)
                #plot shuffled data sem
                ax1.fill_between(xaxis,chan1_shuff_5, chan1_shuff_95, color="grey", alpha = 0.1 )
                #ax6.get_yaxis().set_visible(False)
            
        
        if save == 1:
            filename = f'All ROIs_TrailAve_RunAve{run_win}_{Channel}_{Filebase}_Shuff={shuff}_{data}_{i}.png'
            dir_path = os.path.join(r'Dinghao\code_dinghao\dLight', Filebase[0:5], 'Plot_'+mode+'_fullshuff' , data, Filebase)
            #dir_path = os.path.join(r'Z:\Jingyu\2P_Recording\\', Filebase[0:5],date, session, mode, 'Plot')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            path = os.path.join (dir_path, filename)
            fig.savefig(path, bbox_inches='tight')
            plt.cla()
            plt.clf()               
       
        if save == 0:
            plt.show()

#%% Load data from hkl or .mat
Filebase='AC918-20231017-00'
# Filebase = 'AC912-20230318-04'
Channel = '2Chan'
date =  Filebase[:-3]
session = Filebase[-2:]
mode = r'NicoCaimen'


#load data without shuffle
chan1 = os.path.join(r'Z:\Jingyu\2P_Recording\\', Filebase[0:5],date, session, mode, 'Channel1', f'A{Filebase[2:6]}{Filebase[6:]}_DataStructure_mazeSection1_TrialType1.mat')
chan2 = os.path.join(r'Z:\Jingyu\2P_Recording\\', Filebase[0:5],date, session, mode, 'Channel2', f'A{Filebase[2:6]}{Filebase[6:]}_DataStructure_mazeSection1_TrialType1.mat')

chan1, Beh=load(chan1)
chan2, Beh=load(chan2)


# dinghao 
# indices in Beh were upsampled to 500 Hz
recpath = 'Z:\Jingyu\2P_Recording\AC926\AC926-20240305\02\RegOnly\suite2p\plane0\K_4_p_2_decay_time_6\mock_suite2p\F.npy'
F = np.load(recpath)

starts = Beh['starts']
starts = [int(s) for s in starts if int(s)/500*30<5000]  # for this recording only 
tot_trial = len(starts)
tot_roi = F.shape[0]
trial_F = np.zeros((tot_roi, 3*40-2))
for roi in range(tot_roi):
    temp = []
    for trial in range(tot_trial):
        start_frame = int(starts[trial]/500*30)
        temp.append(normalise(moving_average(F[roi, start_frame:start_frame+4*30])))  # start + 4 seconds
    trial_F[roi, :] = np.mean(temp, axis=0)

max_pt = {}  # argmax for conts for all rois
for i in range(tot_roi):
    max_pt[i] = np.argmax(trial_F[i,:])
def helper(x):
    return max_pt[x]
ord_ind = sorted(np.arange(tot_roi), key=helper)

im_max = np.zeros((tot_roi, 30*4-2))
for i, ind in enumerate(ord_ind): 
    im_max[i,:] = trial_F[ind,:]


delay  = Beh['delay']*10**-6


# data processing

#seperate good trials

firstLicks =[]; lastLicks = []; lick_freq_max=[]
for t in range(tot_trial):
    licks = Beh['licks'][t]
    trl_len = (chan1['dFFgf'][t][:,0]).shape[0]
    
    if licks is not None:
        firstLicks.append(licks[0])
        lastLicks.append(licks[-1])       
        lick_freq_max.append(max(np.histogram(licks, range=(0,trl_len), bins = int(trl_len/50))[0])) #lick times/0.1s
    else: 
        firstLicks.append(0)
        lastLicks.append(0)

Beh.update({'firstLicks':firstLicks, "lastLicks": lastLicks, 'lick_freq_max': lick_freq_max})

# good_trials = [i for i in range(len(firstLick)) if firstLick[i]!= 0 and firstLick[i] >=  0.5*delay*10**3]
# bad_trials = [i for i in range(len(firstLick)) if firstLick[i]!= 0 and firstLick [i] <  0.5*delay*10**3]
sort_lick = sorted(range(len(firstLicks)), key=lambda k: firstLicks[k]) #sort trials with firtslicks
bad_trials = sort_lick[0:int(tot_trial*0.3)] #equals to early_lick trial
good_trials = sort_lick[int(tot_trial*0.7):]#equals to late_lick trial

# temporal
Ave_trial(chan1, 'dFFgf', 1, 0, min_length, tot_ROI, tot_trial)
Ave_trial(chan2, 'dFFgf', 1, 0, min_length, tot_ROI, tot_trial)
run_win = 'None'
Plot_ave('dFFgf', shuff=0, save=0)