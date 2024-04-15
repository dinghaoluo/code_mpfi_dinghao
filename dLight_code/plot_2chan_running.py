# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:59:41 2023
Modified on Thu 26 Oct

@author: Jingyu Cao, Dinghao Luo

"""


#%% imports
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.stats
import mat73
import os
import math
import time
import random

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% functions
def load(filename, align, mode, chan): 
    
    """
    Parameters
    ----------
    filename : str
        eg. 'A058-20230420-04'
    align : str
        'Rew', 'Run', 'Cue'
    mode : str
        'axons_v2.0'
    chan : str     
        'Channel1' or 'Channel2'

    Returns
    -------
    list
        DESCRIPTION.

    """

    date =  filename[:-3]
    session = filename[-2:]

    # get clulist first
    pathclulist = os.path.join(r'Z:\Jingyu\2P_Recording\\', filename[0:4], date, session, mode, chan, f'A{filename[1:5]}{filename[5:]}_DataStructure_mazeSection1_TrialType1.mat')
    clulist = mat73.loadmat(pathclulist)['cluList']['localClu']
    
    # get main data
    path = os.path.join(r'Z:\Jingyu\2P_Recording\\', filename[0:4],date, session, mode, chan, f'A{filename[1:5]}{filename[5:]}_DataStructure_mazeSection1_TrialType1_align{align}_msess1.mat')
    mat = mat73.loadmat(path)
    mat_real = mat['trials' + align]
    
    tot_trial = len(mat_real['dFF'])
    # extract dFF
    # dFFSM = []
    # for i in range (0, tot_trial):
    #     dFFSM.append(mat_real['dFF'][i])   
    
    # extract dFFgf
    dFFgf = []
    for i in range (1, tot_trial):
        dFFgf.append(np.concatenate((mat_real['dFFBefGF'][i], mat_real['dFFGF'][i]), axis=0)) #get -3s data 'dFFBfGF
        
    # # extract dFFSM
    # dFFSM = []
    # for i in range (1, tot_trial):
    #     dFFSM.append(np.concatenate((mat_real['dFFBef'][i], mat_real['dFF'][i]), axis=0)) #get -3s data 'dFFBfGF
    
    #extract Fc
    # Fc=[]
    # for i in range (0, tot_trial):
    #     Fc.append(mat_real['F'][i])
    
    #extract behaviour 
    lick = []
    for i in range (1, tot_trial):
        lick.append(mat_real['lickLfpInd'][i])
    
    speed = [] #mm/sec
    for i in range (1, tot_trial):
        speed.append(mat_real['speed_MMsec'][i])
    # Rew = []
    # for i in range (0,len(mat_real)):
    #     Rew.append(mat_real[i]['pumpLfpInd'])
    
    # start = []
    # for i in range (0,len(mat_real)):
    #     start.append(mat_real[i]['lfpIndStart'])
    
    return {'clulist':clulist, 'dFFgf':dFFgf, 'lick':lick, 'speed': speed}


def get_roi(ROIid):
    
    return chan1['clulist'].astype(int).tolist().index(ROIid+1)


def get_id(ROI_seq):
    
    return int(chan1['clulist'][ROI_seq])-1


def zero_padding(vector, max_length):
    new_vector = np.zeros(max_length)
    l = len(vector)
    new_vector[:l] = vector
    
    return new_vector


def rollingmean (data, w, axis): # a =array, w=window_size, axis = 0 for Z
    from scipy.ndimage import uniform_filter1d
    lst=[]
    #'reflect' mode extend input array beyond boundaries by reflecting the edge of the last pixel, thus input frames = output
    # to avoide this, get slice for part of the array
    hw = w//2
    
    for t in range (len(data)):
        frames = data[t].shape[axis]-w+1
        indexer = [slice(None) for i in range (data[t].ndim)]
        indexer[axis] = slice(hw, hw + frames)
        array_tri = uniform_filter1d(data[t],w,axis=axis)[tuple(indexer)]
        lst.append(array_tri)
    return lst


def normalise(vector):
    maxv = max(vector)
    minv = min(vector)
    new_vector = np.zeros(len(vector))
    for i in range(len(vector)):
        new_vector[i] = (vector[i] - minv) / (maxv - minv)
    
    return new_vector


def shuffle(chan, data, times): #new shuffle function v2.6
    
    tot_ROI = chan[data][1].shape[1]
    tot_trial= tot_trial = len(chan[data])
    min_length = min((chan[data][t][:,0]).shape[0] for t in range(tot_trial))
    
    print ('shuffling starts')
    start_time = time.time()
    

    shuff_95 = np.zeros((min_length, tot_ROI))
    shuff_5 = np.zeros((min_length, tot_ROI))
    shuff_mean = np.zeros((min_length, tot_ROI))
    
    print_thresholds = list(reversed([tot_ROI//n for n in np.arange(2,6)]))

    for r in range(tot_ROI):
        if r in print_thresholds:
            print('{}% done.'.format((print_thresholds.index(r)+1)*20))
        roi_shuffle_all = np.zeros((min_length, times))
        for i in range(times):
            shuffle_trials = [np.roll(chan[data][t][:,r], -random.randint(0, int(len(chan[data][t])/2)))[:min_length] for t in range(tot_trial)]
            each_shuffle = np.mean(shuffle_trials, axis=0)
        
            roi_shuffle_all[:,i] = each_shuffle
        
        shuff_95[:,r] = np.percentile(roi_shuffle_all, 90, axis=1)
        shuff_5[:,r] = np.percentile(roi_shuffle_all, 10, axis=1)
        shuff_mean[:,r]=np.mean(roi_shuffle_all, axis=1)
                
    print('shuffling ends ({} seconds)\n'.format(round(time.time() - start_time, 2)))
        
    chan.update({data+'_shuff': shuff_mean, data+'_shuff_95':shuff_95 , data+'_shuff_5':shuff_5 }) 


def ave_trial(chan, data, sem): #get trial_ave for each roi with SEM 

    tot_ROI = chan[data][1].shape[1]
    tot_trial= tot_trial = len(chan[data])
    min_length = min((chan[data][t][:,0]).shape[0] for t in range(tot_trial))

    #list1  =chan1[data]
    list1_ave_ROIs = []
    list1_sem_ROIs = []

    for r in range(tot_ROI):
        
        roi_list1_trials = np.zeros([min_length, tot_trial])
      
        for t in range(tot_trial):
            roi_list1_trials[:,t] = chan[data][t][:,r][:min_length]
        
        list1_ave = np.mean(roi_list1_trials, axis = 1)
        list1_ave_ROIs.append(list1_ave)
        
        chan.update({data+'_ave': list1_ave_ROIs})
        
        if sem == 1:
            list1_sem = scipy.stats.sem(roi_list1_trials, axis = 1)
            list1_sem_ROIs.append(list1_sem)
            
            chan.update({data+'_sem': list1_sem_ROIs})


def get_rSD(chan1, chan2):
    tot_ROI = chan1['dFFgf'][1].shape[1]
    tot_trial = len(chan1['dFFgf'])
    chan1_rSD = []
    chan2_rSD = []
    for i in range(tot_ROI):    
        chan1_session = np.hstack([chan1['dFFgf'][t][:,i] for t in range(tot_trial)])
        chan1_roi_rSD = scipy.stats.median_abs_deviation(chan1_session)/0.6745 #david tank robust SD 2021 nature
        chan1_rSD.append(chan1_roi_rSD)
        chan2_session = np.hstack([chan2['dFFgf'][t][:,i] for t in range(tot_trial)])
        chan2_roi_rSD = scipy.stats.median_abs_deviation(chan2_session)/0.6745
        chan2_rSD.append(chan2_roi_rSD)
    chan1.update({'rSD':chan1_rSD})
    chan2.update({'rSD':chan2_rSD})


def stdThreshold(chan1, chan2, para):
    tot_ROI = chan1['dFFgf'][1].shape[1]
    tot_trial = len(chan1['dFFgf'])
    chan1_dFFSM= []
    chan2_dFFSM= []
    for t in range(tot_trial):
        chan1_dFFSM_rsd = np.empty(chan1['dFFgf'][t].shape)
        chan2_dFFSM_rsd = np.empty(chan1['dFFgf'][t].shape)
        for r in range(tot_ROI):

            temp1 = chan1['dFFgf'][t][:,r].copy()
            idx = np.where(np.abs(temp1)<para*chan1['rSD'][r])
            temp1[idx]=0
            chan1_dFFSM_rsd[:,r]=temp1 
            
            temp2 = chan2['dFFgf'][t][:,r].copy()
            idx = np.where(np.abs(temp2)<para*chan2['rSD'][r])
            temp2[idx]=0
            chan2_dFFSM_rsd[:,r]=temp2 
            
        chan1_dFFSM.append(chan1_dFFSM_rsd)
        chan2_dFFSM.append(chan2_dFFSM_rsd)
        
    chan1.update({'dFFSM_'+str(para)+'std':chan1_dFFSM})
    chan2.update({'dFFSM_'+str(para)+'std':chan2_dFFSM})


#%%plot
def plot_ave(data, shuff, save, filename):
    plots = 50 #plots per figure
    fig_num = math.ceil(tot_ROI/plots)
    
    #delay = Beh['delay']*10**-6
    
    for i in range(fig_num):
        roi_col = 5
        roi_row = math.ceil(plots/roi_col)
        
        min_length = len(chan1[data+'_ave'][0])-2*500
        fig = plt.figure(i, figsize=[roi_col*5, roi_row*3.2])
        #fig.subplots_adjust(top=1.5)
        fig.suptitle(f'All ROIs_TrialAve__align{align}_Runave{run_win}_{channel}-{filename}_Shuff={shuff}', size=15)
        fig.subplots_adjust(top=0.97)
    
    
        for r in range(plots*i, min(plots*(i+1), tot_ROI)):
            xaxis = np.arange(-1*500, min_length-1*500)/500
            plot_pos = np.arange(1, min(plots*(i+1),tot_ROI)+1)
            ax1 = fig.add_subplot(roi_row, roi_col, plot_pos[r-plots*i])
            subtitle = 'ROI' + str(int(chan1['clulist'][r])) + '(' + str(r) + ')'
            
            chan1_mean = chan1[data+'_ave'][r][2*500:]
            # chan2_mean = chan2[data+'_ave'][r][2*500:]
            
            chan1_sem = chan1[data+'_sem'][r][2*500:]
            # chan2_sem = chan2[data+'_sem'][r][2*500:]
            
            y_max1 = max(0.1,max(chan1_mean+chan1_sem))+0.01
            y_min1 = min(chan1_mean-chan1_sem)-0.01
            
            
            ax1.title.set_text(subtitle)
            ax1.set(ylabel= 'chan1'+data, xlabel='time (s)',
                    xlim=(-1, min_length/500-1), ylim=(y_min1, y_max1))
           

            ln, = ax1.plot(xaxis, chan1_mean, color='green')
            # ax1.plot(xaxis, chan2_mean, color="#800080", alpha=.5)
            
            ax1.fill_between(xaxis,chan1_mean-chan1_sem, chan1_mean+chan1_sem, color='green', alpha = 0.1)
            # ax1.fill_between(xaxis,chan2_mean-chan2_sem, chan2_mean+chan2_sem, color="#800080", alpha = 0.05)
            
            
            if shuff ==1:
            
                chan1_shuff_mean = chan1[data+'_shuff'][:,r][2*500:]
                chan1_shuff_95 = chan1[data+'_shuff'+'_95'][:,r][2*500:]
                chan1_shuff_5 = chan1[data+'_shuff'+'_5'][:,r][2*500:]
                #plot shuffled data mean
                shufln, = ax1.plot(xaxis, chan1_shuff_mean, color="grey", linewidth=0.5)

                #plot shuffled data 5-95%
                ax1.fill_between(xaxis,chan1_shuff_5, chan1_shuff_95, color="grey", alpha = 0.1 )
                
            ax1.legend([ln, shufln], ['dLight 3.0', 'shuffled'], frameon=False)
            
            for p in ['top', 'right']:
                ax1.spines[p].set_visible(False)
            
        fig.tight_layout()
        plt.show()
        
        if save == 1:
            fig.savefig('Z:\Dinghao\code_dinghao\dLight\{}_{}.png'.format(filename, i), bbox_inches='tight')
        
        plt.close(fig)


#%%test 
sessions = [
    'A054-20230310-03',
    'A054-20230310-06',
    'A054-20230311-04',
    'A054-20230314-04',
    'A054-20230314-06',
    
    'A058-20230413-06',
    'A058-20230420-04',
    'A058-20230421-02'
    ]


for sessname in sessions:
    print(sessname)
    
    channel = '2Chan'
    
    align = 'Run'
    mode = 'axons_v2.0'
    data = 'dFFgf'
    
    chan1 = load(sessname, align, mode, chan = 'Channel1')
    chan2 = load(sessname, align, mode, chan = 'Channel2')
    
    tot_trial = len(chan1[data]) -1 #lost first trial because of aligenment
    tot_ROI = chan1[data][1].shape[1]
    
    run_win = 'None'
    
    ave_trial(chan1, data, sem=1)
    ave_trial(chan2, data, sem=1)
    
    shuffle(chan1, data, 200)
    shuffle(chan2, data, 200)
    
    plot_ave(data, 1, 1, sessname)
    
# filename = 'A054-20230314-04'
# channel = '2Chan'
    
# align = 'Run'
# mode = 'axons_v2.0'
# data = 'dFFgf'
    
# chan1 = load(filename, align, mode, chan = 'Channel1')
# chan2 = load(filename, align, mode, chan = 'Channel2')
    
# tot_trial = len(chan1[data]) -1 #lost first trial because of aligenment
# tot_ROI = chan1[data][1].shape[1]
    
# run_win = 'None'
    
# ave_trial(chan1, data, sem=1)
# ave_trial(chan2, data, sem=1)
    
# shuffle(chan1, data, 500)
# shuffle(chan2, data, 500)
    
# plot_ave(data, shuff=1, save=1)