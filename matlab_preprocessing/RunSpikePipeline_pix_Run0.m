
function RunSpikePipeline_pix_Run0()
    % Runs all spike-sorting pipeline functions
    % Dinghao 23 Jan, 2023
    
    % remember to change directory to rec folder before use
    
    % DIRECTORY ONLY WORKS WITH DINGHAO'S RECORDINGS
    % CHANGE DIRECTORY IF ERROR
    
    % read recording paths
    RecordingList_dinghao;
    paths = listRecordingsActiveLickPathHPCLCOpt(end-1:end, :);
    files = listRecordingsActiveLickFileNameHPCLCOpt(end-1:end, :);
    tot_rec = size(paths,1);
    
    for i = 1:tot_rec
        pathname = paths(i,:);
        cd(pathname);
        
        filename = files(i,1:17);
        disp(filename);
    
        fullfilename = [filename '_DataStructure_mazeSection1_TrialType1'];
        fullpath = ['Z:\Dinghao\MiceExp\ANMD' filename(2:5) '\' filename(1:14)];

%         % generate B.mat, BTDT.mat, -whl.mat, .NeuronQuality.mat,
%         % _BehavElectrDataLFP.mat, _BehavElectrDataLFP_CCG.mat, _eeg_1250Hz.mat
        GenerateBehavElectroDataStructures_smTrMPFIv2_opto_pix(fullpath, filename);
% 
%         % generate _DataStructure_mazeSection1_TrialType1.mat
%         % DOUBLE CHECK STIM PULSE NUMBER
        GetTrials_smTrMPFI(filename,1,1,0,0);
% 
%         % generate processed Info, ext, im, runSpeed, FR_Run0, Depth,
%         % convSpikesTime9p6ms, convSpikesDist20mm, PeakFR20mm, ThetaPhaseH
%         % ThetaPhaseL, Concatsp, CCG, ThetaMod, SpInfo, FieldWidthLR_20mm_L
%         % burstAll_THH, burstAll_THL
        disp([newline 'PROCESSING'])
        ProcessingMice_smTr('./',fullfilename,0);

        % generate processed thetaPower, alignRun_msess, alignRew_msess,
        % alignCue_msess, alignCueOff_msess, alignedSpikesPerNPerT_msess,
        % convSpikesAligned_msess, PeakFRAligned_msess,
        % thetaPhaseOverTimeligned_msess, lickDist_msess, runSpeedDist_msess,
        % FRAlignedRun_msess, ThetaPhaseAligned files, and rasters
        disp([newline 'ALIGNING'])
        ProcessingAlignedWithStim('./',fullfilename,0,1,0,3);

        % control only 
        disp([newline 'PROCESSING CONTROL ONLY'])
        ProcessingMice_smTrCtrlOnly('./',fullfilename,0,1)


        % tagging-related functions 
    %     stimEffect_NewData_MPFI('./',fullfilename);
    %     plotStimRasterWrapper('./',fullfilename, -1);
    
    end
end

