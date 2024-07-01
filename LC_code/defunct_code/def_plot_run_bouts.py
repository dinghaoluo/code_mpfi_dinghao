# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:09:35 2023

plots run-bout identification figure for one or more animals

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
import mat73 
import pandas as pd


#%% gaussian kernal for speed smoothing
samp_freq = 1250
gx_speed = np.arange(-50, 50, 1)  # xaxis for Gaus
sigma_speed = samp_freq/100
gaus_speed = [1 / (sigma_speed*np.sqrt(2*np.pi)) * 
              np.exp(-x**2/(2*sigma_speed**2)) for x in gx_speed]


#%% MAIN
file_name = 'A049r-20230120-04'
file_path = r'Z:\Dinghao\MiceExp\ANMD'+file_name[1:5]+'\\'+file_name[:14]+'\\'+file_name[:17]+'\\'+file_name[:17]
run_bout_path = r'Z:\Dinghao\code_dinghao\run_bouts'
run_bout_path = run_bout_path+'\\'+file_name[:17]+'_run_bouts_py.csv'
run_bout_table = pd.read_csv(run_bout_path)
    
behave_lfp_path = file_path+'_BehavElectrDataLFP.mat'
beh_lfp = mat73.loadmat(behave_lfp_path)
tracks = beh_lfp['Track']; laps = beh_lfp['Laps']
pumpLfp = laps['pumpLfpInd']
pumpLfp = [np.mean(pump) for pump in pumpLfp]
speed_MMsec = max(tracks['speed_MMsec'])
    
run_start_lfp = run_bout_table.iloc[:,1]


for i in range(len(run_bout_table)):      
    run_lfp = run_bout_table.run_start_lfp(i) - 5*1250 : run_bout_table.run_start_lfp(i) + 5*1250;
    
    pump_run_bout_i = pumpLfp(ismember(pumpLfp, run_lfp_indices));
        close_run_bouts = run_start_lfp_indices(ismember(run_start_lfp_indices, run_lfp_indices) & run_start_lfp_indices ~= run_bout_table.run_start_lfp(i));
              
        acc_ro = run_bout_table.acc_run_onset(i);
        prec_pause = run_bout_table.precede_pause_length_sec(i);
        mean_speed = run_bout_table.mean_speed_run(i);
        
        CreateFig();
        hold on;
        plot(run_lfp_indices/1250, speed_MMsec(run_lfp_indices));
        start_run = (run_lfp_indices(1) + 5*1250)/1250;     
        plot([start_run, start_run], [0 max(speed_MMsec(run_lfp_indices))], 'r--');         
        pump_y = 10+max(speed_MMsec(run_lfp_indices));
        scatter(pump_run_bout_i/1250, pump_y*ones(size(pump_run_bout_i)), 'b.');
        for j = 1:size(close_run_bouts)
            plot([close_run_bouts(j), close_run_bouts(j)]/1250, [0 max(speed_MMsec(run_lfp_indices))], 'm--');   
        end
        
        title_string = string(file_name(1:16)) +...
                       "\nmean speed: " + num2str(mean_speed) + ...
                       "\nprec pause: " + num2str(prec_pause) + ...
                       "\nacc_ro: " + num2str(acc_ro);
        title(char(compose(title_string)));
        
        save_path = [save_path_base file_name(1:16) '_run_bout' num2str(i) '.png'];
        saveas(gcf , save_path);
        close all;

save_path_base = 'Z:\Raphael_tests\Code\matlabAnalysisRaphi\RiseDownOffTargetRunBouts\verify_run_bout_plots\';
save_path_base = [save_path_base file_name(1:16) '\'];