function RunSpikePipeline(filename)
    % Runs all spike-sorting pipeline functions
    % Dinghao 23 Jan, 2023
    
    % remember to change directory to rec folder before use
    
    addpath Z:\Dinghao\code 
    pathmatlab_smTr_dinghao();
    
    % DIRECTORY ONLY WORKS WITH DINGHAO'S RECORDINGS
    % CHANGE DIRECTORY IF ERROR
%     filename = 'A049r-20230104-04';
    fullfilename = [filename '_DataStructure_mazeSection1_TrialType1'];
    fullpath = ['Z:\Dinghao\MiceExp\ANMD' filename(2:5) '\' filename(1:14)];
    
    % generate B.mat, BTDT.mat, -whl.mat, .NeuronQuality.mat,
    % _BehavElectrDataLFP.mat, _BehavElectrDataLFP_CCG.mat, _eeg_1250Hz.mat
    GenerateBehavElectroDataStructures_smTrMPFIv2_opto(fullpath, filename);
    
    % generate _DataStructure_mazeSection1_TrialType1.mat
    % DOUBLE CHECK STIM PULSE NUMBER
    GetTrials_smTrMPFI(filename,1,1,0,0);
    
    % generate processed Info, ext, im, runSpeed, FR_Run0, Depth,
    % convSpikesTime9p6ms, convSpikesDist20mm, PeakFR20mm, ThetaPhaseH
    % ThetaPhaseL, Concatsp, CCG, ThetaMod, SpInfo, FieldWidthLR_20mm_L
    % burstAll_THH, burstAll_THL
    disp([newline 'PROCESSING'])
    ProcessingMice_smTr('./',fullfilename,0);
    
    % generate processed thetaPower, alignRun_msess, alignRew_msess,
    % alignCue_msess, alignCueOff_msess, alignedSpikesPerNPerT_msess,
    % convSpikesAligned_msess, PeakFRAligned_msess,
    % thetaPhaseOverTimeligned_msess, lickDist_msess, runSpeedDist_msess,
    % FRAlignedRun_msess, ThetaPhaseAligned files, and rasters
    disp([newline 'ALIGNING'])
    ProcessingAlignedWithStim('./',fullfilename,0,1,0,3);
    
    % tagging-related functions 
    stimEffect_NewData_MPFI('./',fullfilename);
    plotStimRasterWrapper('./',fullfilename, -1);
    
    
    % control only 
    ProcessingMice_smTrCtrlOnly('./',fullfilename,0,1)
    
end