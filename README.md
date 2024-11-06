# code_dinghao_mpfi
my scripts at MPFI; unless otherwise specified, I am the author of all the scripts in this repo

## directory tree
├── **HPC_code**
│   ├── HPCLC_all_rasters.py
│   ├── HPCLC_all_train.py
│   ├── HPCLC_all_train_cue.py
│   ├── HPCLC_clu_list.py
│   ├── HPCLC_place_cell_profiles.py
│   ├── HPCLC_pyract_profiles.py
│   ├── HPCLC_pyract_single_cell_profiles.py
│   ├── HPCLC_raster_first_lick_ranked.py
│   ├── HPCLC_sess_pyr_profiles.py
│   ├── HPCLC_theta_stim.py
│   ├── HPCLC_waveform_all.py
│   ├── HPCLC_waveform_proc.py
│   ├── HPCLCterm_all_rasters.py
│   ├── HPCLCterm_all_train.py
│   ├── HPCLCterm_clu_list.py
│   ├── HPCLCterm_place_cell_profiles.py
│   ├── HPC_UMAP_single_trial_traj_interactive.ipynb
│   ├── HPC_all_1st_lick_ordered.py
│   ├── HPC_population_v_licks.py
│   ├── HPC_population_v_licks_poisson.py
│   ├── HPC_population_v_licks_poisson_example_session.py
│   ├── HPC_session_PCA_traj.py
│   ├── HPC_session_UMAP_traj.py
│   ├── HPC_single_trial_UMAP_traj.py
│   ├── all_acg.py
│   ├── all_acg_baseline.py
│   ├── **behaviour**
│   │   ├── lick_dist_comp_HPC_LC_stim.py
│   │   ├── lick_dist_comp_HPC_LCterm_stim.py
│   │   ├── lick_time_comp_HPC_LC_stim.py
│   │   └── lick_time_comp_HPC_LCterm_stim.py
│   ├── **defunct_code**
│   │   ├── def_HPCLC_all_stim_stimcont_pyr_only_sig.py
│   │   ├── def_HPCLCterm_all_stim_stimcont_pyr_only_fw.py
│   │   ├── def_HPCLCterm_all_stim_stimcont_pyr_only_sig.py
│   │   ├── def_HPC_all_stim_stimcont_pyr_only_sensitivity.py
│   │   └── def_all_pyr_rate.py
│   ├── **figure_code**
│   │   ├── HPCLC_plot_all_rasters.py
│   │   ├── plot_all_pyr_heatmap_dist.py
│   │   ├── plot_all_pyr_heatmap_time_rew.py
│   │   ├── plot_all_pyr_heatmap_time_run.py
│   │   ├── plot_lick_profile_to_pumps_HPCLC.py
│   │   └── plot_lick_profile_to_pumps_HPCLCterm.py
│   ├── lick_history_dependency.py
│   ├── naive_bayesian_decoding_conf_matrix.py
│   ├── naive_bayesian_decoding_proba.py
│   ├── naive_bayesian_decoding_proba_pyract.py
│   ├── naive_bayesian_decoding_proba_pyrinh.py
│   ├── pc_mc_overlaps.py
│   ├── pc_proportion_comp.py
│   ├── **sequence**
│   │   ├── HPCLC_plot_sequence.py
│   │   ├── HPCLC_plot_sequence_dist.py
│   │   ├── HPCLC_plot_super_sequence.py
│   │   ├── HPCLCterm_plot_sequence.py
│   │   └── HPCLCterm_plot_sequence_dist.py
│   ├── **stim_baseline**
│   │   ├── HPC_all_stim_baseline_int_only.py
│   │   ├── HPC_all_stim_baseline_pyr_only.py
│   │   └── HPC_all_stim_baseline_pyr_only_PCA.py
│   ├── **stim_ctrl**
│   │   ├── HPCLC_all_stim_ctrl_int_only.py
│   │   ├── HPCLC_all_stim_ctrl_population_deviation_poisson.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_PCA.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_heatmap.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_ordered.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_rasters.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_rasters_blowup.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_rasters_signif_only.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_sig_MI.py
│   │   ├── HPCLC_all_stim_ctrl_pyr_only_spatial_info.py
│   │   ├── HPCLC_stim_ctrl_all_int_profiles.py
│   │   ├── HPCLC_stim_ctrl_all_pyr_profiles.py
│   │   ├── HPCLC_stim_ctrl_all_pyr_profiles_single_sess.py
│   │   ├── HPCLCterm_all_stim_ctrl_int_only.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only_heatmap.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only_ordered.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only_rasters.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only_rasters_blowup.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only_rasters_signif_only.py
│   │   ├── HPCLCterm_all_stim_ctrl_pyr_only_sig_MI.py
│   │   ├── HPCLCterm_stim_ctrl_all_pyr_profiles_single_sess.py
│   │   ├── all_HPCLC_HPCLCterm_stim_ctrl_int_only_sig.py
│   │   ├── all_HPCLC_HPCLCterm_stim_ctrl_pyr_only_sig.py
│   │   └── all_mod_depth_comp.py
│   └── **stim_effect**
│       ├── HPCLC_rasters_stim_nonstim.py
│       └── HPC_stim_aligned.py
├── **LC_code**
│   ├── WT_waveform_proc.py
│   ├── all_UMAP.py
│   ├── all_acg.py
│   ├── all_acg_baseline.py
│   ├── all_cell_properties.py
│   ├── all_earlyvlate_RO_peak.py
│   ├── all_earlyvlate_RO_peak_population.py
│   ├── all_earlyvlate_RO_peak_population_lick2pump.py
│   ├── all_earlyvlate_RO_peak_population_med3licks.py
│   ├── all_earlyvlate_RO_peak_population_raster.py
│   ├── all_earlyvlate_RO_peak_raster.py
│   ├── all_goodvbad_RO_peak.py
│   ├── all_goodvbad_RO_peak_bef.py
│   ├── all_lick_sensitive_activity_1st_lick.py
│   ├── all_lick_sensitive_summary.py
│   ├── all_lick_sensitive_summary_shuf.py
│   ├── all_neu_activity_v_1st_lick.py
│   ├── all_neu_activity_v_1st_lick_shuf.py
│   ├── all_raster_cue_rew_run_lasttocurtr.py
│   ├── all_raster_last_rew_ordered.py
│   ├── all_raster_lick_aligned.py
│   ├── all_raster_lick_ordered.py
│   ├── all_raster_lick_ordered_earlyvlate_only.py
│   ├── all_raster_lick_ordered_raster_only.py
│   ├── all_raster_lick_reward_sensitivity.py
│   ├── all_raster_rew_ordered.py
│   ├── all_raster_rew_to_run_ordered.py
│   ├── all_rasters.py
│   ├── all_rovrb_RO_peak.py
│   ├── all_time_warped.py
│   ├── all_train.py
│   ├── all_waveform_all.py
│   ├── all_waveform_proc.py
│   ├── **behaviour**
│   │   ├── 1st_lick_profile.py
│   │   ├── egsess_lick.py
│   │   ├── egsess_lick_passive_raphi.py
│   │   ├── egsess_speed.py
│   │   ├── egsess_speed_passive_raphi.py
│   │   ├── good_perc_comp.py
│   │   ├── lick_dist_comp.py
│   │   ├── lick_dist_comp_HPC_LC_stim.py
│   │   ├── lick_history_dependency.py
│   │   ├── lick_history_dependency_comp.py
│   │   ├── lick_properties.py
│   │   ├── lick_time_comp.py
│   │   ├── plot_cue_start_difference.py
│   │   ├── plot_run_bouts.py
│   │   └── plot_single_trial_example.py
│   ├── **defunct_code**
│   │   ├── def_all_acg_pca.py
│   │   ├── def_all_ccg.py
│   │   ├── def_all_goodvbad_clustered.py
│   │   ├── def_all_rovrb_clustered.py
│   │   ├── def_all_speed_score.py
│   │   ├── def_all_stim_effects_avg.py
│   │   ├── def_beh_eg_licks.py
│   │   ├── def_goodvbad_RO_peak.py
│   │   ├── def_plot_run_bouts.py
│   │   ├── def_plot_single_cell_property_UMAP_assist.py
│   │   ├── def_plot_single_cell_property_UMAP_interactive_image.py
│   │   ├── def_plot_single_cell_property_tagged.py
│   │   ├── def_session_anal.py
│   │   ├── def_stim_trial_only.py
│   │   ├── def_tag_waveform_example.py
│   │   ├── def_tagged_burst_badtrial.py
│   │   ├── def_tagged_cluster_waveform.py
│   │   ├── def_tagged_clustering_from_all.py
│   │   ├── def_tagged_clustering_hierarchical.py
│   │   ├── def_tagged_clustering_rate.py
│   │   ├── def_tagged_goodvbad.py
│   │   ├── def_tagged_goodvbad_clustered.py
│   │   ├── def_tagged_lickvburst.py
│   │   ├── def_tagged_narrvbrd.py
│   │   ├── def_tagged_rewardvburst.py
│   │   ├── def_tagged_rovrb_clustered.py
│   │   ├── def_tagged_rovrb_trough.py
│   │   ├── def_tagged_single_trial_example_cue_RO_diff.py
│   │   ├── def_waveform_anal.py
│   │   ├── def_waveform_comp.py
│   │   └── **hierarchical_clustering**
│   │       ├── all_cluster_waveform.py
│   │       └── all_clustering_hierarchical.py
│   ├── **ephys_opto**
│   │   ├── all_stim_rasters.py
│   │   ├── nontagged_stim_rasters.py
│   │   ├── stim_effect_spikes.py
│   │   └── tagged_stim_rasters.py
│   ├── **figure_code**
│   │   ├── plot_RO_v_nonRO.py
│   │   ├── plot_all_clustered_acg_heatmap.py
│   │   ├── plot_avg_acg_tagged.py
│   │   ├── plot_demo_tonic_phasic.py
│   │   ├── plot_eg_tagged_cell_wf.py
│   │   ├── plot_example_train.py
│   │   ├── plot_heatmap_ROpeaking.py
│   │   ├── plot_heatmap_argmax.py
│   │   ├── plot_proportion_bar_graph.py
│   │   ├── plot_single_cell_ACG.py
│   │   ├── plot_single_cell_property.py
│   │   ├── plot_single_cell_property_tagged.py
│   │   ├── plot_single_cell_raster_RO_aligned.py
│   │   ├── plot_single_cell_waveform.py
│   │   ├── plot_tagged_example_goodvbad_raster.py
│   │   ├── plot_tagged_responses.py
│   │   ├── putative_example_cell.py
│   │   └── tagged_example_cell.py
│   ├── plot_trials_LC.py
│   ├── tag_waveform_all.py
│   ├── tag_waveform_proc.py
│   ├── tag_waveform_proc_notnorm.py
│   ├── tagged_acg_pca.py
│   ├── tagged_goodvbad.py
│   ├── tagged_goodvbad_RO_peak.py
│   ├── tagged_lick_RO_peak_ordered.py
│   ├── tagged_rovrb_RO_peak.py
│   ├── tagged_speedvrate.py
│   ├── tagged_train.py
│   ├── tagged_train_alignedRew.py
│   ├── tagged_waveform_classify.py
│   ├── tagged_waveform_pca.py
│   ├── tagging_latency.py
│   └── **utils**
│       ├── RO_peak_detection.py
│       ├── paramC.py
│       └── single_unit.py
├── README.md
├── **caiman_code** (not in use; mostly by [Nico Spiller](https://github.com/nspiller))
│   ├── 2nd_channel_registration.py
│   ├── Untitled.ipynb
│   ├── batch_cnmf.ipynb
│   ├── cnmf.py
│   ├── utils.py
│   ├── utils_mesmerize.py
│   └── visualize.ipynb
├── **imaging_code**
│   ├── **defunct_code**
│   │   ├── GRABNE_grid_roi.py
│   │   ├── after_suite2p.py
│   │   ├── dLight_all_heatmap.py
│   │   ├── dLight_plot.py
│   │   ├── deepvid_grid_roi.py
│   │   └── grid_extract.py
│   ├── extract_axon_GCaMP.py
│   ├── extract_significant_ROI.py
│   ├── **figure_code**
│   │   ├── plot_lick_profile.py
│   │   ├── plot_lick_profile_to_pumps.py
│   │   └── plot_pooled_heatmap_axon_GCaMP.py
│   ├── plot_directory_tree.py
│   ├── plot_sorted_heatmaps_grids.py
        *plots the heatmaps (unsorted and sorted by argmax) for each session's grid ROIs aligned to run-onsets or rewards*
│   ├── plot_sorted_heatmaps_rois.py
│   ├── run_imaging_pipeline.py
│   ├── **suite2p_code**
│   │   └── registration_roi_extraction.py
│   ├── **tonic_activity**
│   │   ├── tonic_fft.py
│   │   └── whole_session_f_dff.py
│   └── **utils**
│       ├── imaging_pipeline_functions.py
│       ├── imaging_pipeline_main_functions.py
│       ├── imaging_utility_functions.py
│       └── suite2p_functions.py
├── **other_code**
│   ├── log_temperature_humidity.py
│   └── plot_temperature_humidity.py
├── process_behaviour_GRABNE.py
├── process_behaviour_HPCLC.py
├── process_behaviour_HPCLCterm.py
├── process_behaviour_LC.py
├── process_behaviour_axon_GCaMP.py
└── **utils**
    ├── behaviour_functions.py
    ├── common.py
    ├── logger_module.py
    ├── param_to_array.py
    ├── plotting_functions.py
    ├── preprocessing.py
    ├── read_clu.py
    └── txt_processing_functions.py



- **plot_sorted_heatmaps_grids.py** plots the heatmaps (unsorted and sorted by argmax) for each session's grid ROIs aligned to run-onsets or rewards
- **plot_sorted_heatmaps_rois.py** similar to above, but plot the Suite2p-detected ROIs
- **run_imaging_pipeline.py** runs grid ROI or Suite2p ROI trace extraction and alignment to behaviour
### suite2p_code
- **registration_roi_extraction.py** registers and extract ROI traces from each recording in lists of sessions listed in rec_list.py using [the Wang lab version of Suite2p](https://github.com/the-wang-lab/suite2p-wang-lab), fine-tuned by [Nico Spiller](https://github.com/nspiller) and [Jingyu Cao](https://github.com/yuxi62).
### tonic_activity 
- **tonic_fft.py** uses FFT to process entire recording traces after calculating dFF (to get rid of the slow decaying of signal strength due to bleaching and evaporation) in order to look for slow periodicity in the signal
### utils
- **imaging_pipeline_functions.py** contains functions used to process imaging recordings
- **imaging_pipeline_main_functions.py** contains grid ROI and Suite2p ROI extraction and alignment functions; these are what run_imaging_pipeline.py calls directly
- **suite2p_functions.py** contains customised calls to the registration and ROI extraction functions of Suite2p-Wang-Lab

## HPC_code
includes analysis scripts for behaviour and neural data collected from hippocampus recordings
- **all_acg.py** gets ACGs of all the cells throughout entire recording sessions and save them as an .npy file
- **all_acg_baseline.py** is similar to the above script, but gets only ACGs throughout the baseline (pre-stim.) condition, since stimulation may change the ACGs of cells affected by stimulations
- **HPC_all_1st_lick_ordered.py** 
- **HPCLC_all_rasters.py** and **HPCLCterm_all_rasters.py** read spiketime data from behaviour-aligned spiketime files and produce exactly one 1-0 raster array for every recording session
- **HPCLC_all_train.py** and **HPCLCterm_all_train.py** read spiketime data similarly to ...rasters.py, but then convolve the spike train with a 100-ms-sigma Gaussian kernel and produce exactly one spike train array for every recording session
- **HPCLC_clu_list.py** and **HPCLCterm_clu_list.py** generate a list of clu name for every recording session to accelerate later processing
- **HPCLC_place_cell_profiles.py** and **HPCLCterm_place_cell_profiles.py** summarise the place cell indices and number of place cells for each recording session
- **HPC_session_PCA_traj.py** performs PCA on averaged all trials, averaged stim or control trials, and calculate and plot the distances between points on the trajectories
### behaviour
### figure_code
### sequence 
- **HPC_LC_plot_sequence.py** and **HPC_LCterm_plot_sequence.py** plot temporal cell sequences for single sessions
- **HPC_LC_plot_sequence_dist.py** and **HPC_LCterm_plot_sequence_dist.py** plot distance cell sequences for single sessions
### stim_stimcont
**HPCLC_all_stim_stimcont_pyr_only_sig_MI.py** processes stim. vs ctrl. pyramidal profiles based on the HPC_LC_stim_stimcont_diff_profiles_pyr_only.pkl dataframe; it plots the 
### 

## LC_code 
includes analysis scripts for behaviour and neural data collected from locus coeruleus recordings
- **all_acg.py** gets ACGs of all the cells throughout entire recording sessions and save them as an .npy file
- **all_acg_baseline.py** is similar to the above script, but gets only ACGs throughout the baseline (pre-stim.) condition, since stimulation may change the ACGs of hChR2-expressing cells 
- **all_cell_properties.py** is a core function that collects information about each recorded unit from previously produced .npy files and puts them into one big dataframe (LC_all_single_cell_properties.pkl) that contains 'tagged', 'spike_rate' (ctrl trials only), 'peakness' (shuffle-detected, functions located in utils\RO_peak_detection.py), 'putative' (from k-means), 'lick_sensitivity', 'lick_sensitivity_type', the latter 2 of which only appear after running all_lick_sensitive.py as well
- **all_earlyvlate_RO_peak.py** compares single-unit spike rates between early 1st-lick and late 1st-lick trials
- **all_earlyvlate_RO_peak_population.py** compares the population spike rates between early 1st-lick trials and late 1st-lick trials; includes tagged, putative and all LC *Dbh*+ cells
- **all_goodvbad_RO_peak.py** compares the peak spike rate of run-onset peaking *Dbh*+ cells between good and bad trials
- **all_goodvbad_RO_peak_bef.py** compares the slightly-before-peak spike rate of run-onset peaking *Dbh*+ cells between good and bad trials
- **all_lick_sensitive.py** also depends on the dataframe output by all_lick_RO_peak_ordered.py and looks at the neuronal inhibition/activation around the 1st-licks; this function edits the dataframe produced by cell_properties.py to add the 'lick_sensitivity' and 'lick_sensitivity_type' 
- **all_lick_sensitive_activity_1st_lick.py** looks at the time from run-onset to neuronal inhibition of cells that are sensitive to 1st-licks; depends on the dataframe output by all_lick_RO_peak_ordered.py
- **all_raster_cue_rew_run_lasttocurtr.py** does similar things as the functions below, but deals with the spikes before slightly before the cue, the reward and the run-onset of each trial, hence 'lasttocurtr'
- **all_raster_last_rew_ordered.py** is similar to the below function but orders the trials based on the time from run-onset to reward delivery of the last trial
- **all_raster_lick_ordered.py** analyses single cell's response to 1st-licks by ordering the trials based on the time from run-onset to 1st-licks
- **all_raster_lick_reward_sensitivity.py** does not order the trials, but rather determine whether a cell is significantly aligned to rewards or first-licks by comparing the standard deviation of the distributions of transition points around the reward and around the first-licks
- **all_raster_rew_ordered.py** is similar to the above function but orders the trials based on the time from run-onset to reward delivery
- **all_raster_rew_to_run_ordered.py** is similar to the above function but orders the trials based on the time from reward delivery of the last trial to the run-onset of the current trial 
- **all_rasters.py** is a core function that extracts and saves rasters as 0-1 matrices with a structure of trial x time bins
- **all_rovrb_RO_peak.py** compares the peak spike rate of run-onset peaking *Dbh*+ cells between trial run-onsets and spontaneous run-bout-onsets
- **all_time_warped.py** first plots each LC cell's activity time-warped between run-onset and 1st-licks, in order to emphasise these cells' responses to these 2 trial landmarks, and then saves the mean warped activity of all the cells into LC_all_warped.npy
- **all_train.py** is a core function that extracts and saves the spike train (smoothed using a .8-second long Gaussian filter with a sigma of .1 second) into LC_all_info.npy and LC_all_avg_sem.npy, the latter being the mean and sem of the spike rate curve, aligned to run-onsets; it also plots the mean spike rate curve and spike rate heatmap of each cell
- **all_UMAP.py** is my ultimate way to cluster the LC *Dbh*+ cells after trying k-means, hierarchical and PCA; it performs UMAP dimensionality reduction on the ACGs of the LC cells and classify each cell as either putative *Dbh*+ or putative non-*Dbh*+ (besides the tagged *Dbh*+ cells). The reduced ACG points are then clustered using k-means with a k of 2. The results are plotted to LC_all_UMAP_acg(_grey).pdf and LC_all_UMAP_acg_kmeans.pdf and LC_all_UMAP_acg_kmeans_categorised.png and are the basis of the putative *Dbh*+ analyses. This script also produces diagnostic plots and an interactive plot (interactive_UMAP.html) for debugging
- **all_waveform_all.py** depends on all_waveform_proc.py, saving the waveforms to LC_all_waveforms.npy, as well as generating a plot for all of them
- **all_waveform_proc.py** is a core function that extracts the waveform of each spike of a unit from the raw recording file and save them session-by-session
- **tag_waveform_all.py** and **tag_waveform_proc.py** are similar to the above 2 functions, but dedicated to tagged units; **tag_waveform_proc_notnorm.py** processes the same data but without normalisation
- **tagged_acg_pca.py** dimensionality-reduce the PCAs of the tagged *Dbh*+ neurones to see if there are different clusters
- **tagged_goodvbad.py** and **tagged_goodvbad_RO_peak.py** do similar things as all_goodvbad_RO_peak.py but only processes tagged units
- **tagged_lick_RO_peak_ordered.py** is similar to all_lick_RO_peak_ordered.py but only processes tagged units 
- **tagged_rovrb_RO_peak.py** is similar to all_rovrb_RO_peak.py but only processes tagged units 
- **tagged_speedvrate.py** looks at single-trial correlations between a tagged *Dbh*+ cell's spiking activity and the animal's speed
- **tagged_train.py** is similar to all_train.py but only processes tagged cells; **tagged_train_alignRew.py** is similar but align the spiking activity to rewards
- **tagged_waveform_classify.py** categorises tagged *Dbh*+ cells into either broad or narrow
- **tagged_waveform_pca.py** is similar to - tagged_acg_pca.py but runs PCA on the waveforms of tagged units
- **tagging_latency.py** calculates the latency-to-spike for tagged *Dbh*+ units
- **WT_waveform_proc.py** processes waveforms of units from wild-type recordings
### behaviour 
- **1st_lick_profile.py** plots a stair histogram of the first-lick distance and time of all the trials in LC-optotagging sessions
- **egsess_lick.py** plots example mean lick curves
- **egsess_lick_passive_raphi.py** uses Raphael Heldman's data of mice performing the passive task (where water rewards were delivered automatically at the end of each trial) and plots example mean lick curves
- **egsess_speed.py** and **egsess_speed_passive_raphi.py** are similar to the above 2 scripts but for speed 
- **good_perc_comp.py** compares the percentages of good trials between stim. and ctrl. trials
- **lick_dist_comp.py** compares the distance from run-onsets to 1st-licks between stim. and ctrl. trials for LC recordings; **lick_dist_comp_HPC_LC_stim.py** is similar but processes HPC recordings with LC stimulations
- **lick_history_dependency.py** quantifies history-dependency of 1st-lick time based on how the 1st-lick time changes (delta 1st-lick time) on the next trial depending on the current trial's 1st-lick time
- **lick_history_dependency_comp.py** compares the history-dependency of 1st-lick time between stim. and ctrl. trials; needs some more work to be useful (e.g. percentile significance tests, etc.)
- **lick_properties.py** compares the standard deviations of licks between the stim. and the ctrl. trials to see if licks became more concentrated for stimulation trials
- **lick_time_comp.py** is similar to lick_dist_comp.py but analyses the temporal latency from run-onsets to the 1st-licks; it also contains cells that analyse the controls for stim.-v-ctrl. comparison (mean velocity, peak velocity etc.)
- **plot_cue_start_difference.py** plots the cue time and the run-onset time for each trial to demonstrate the separation between these 2 alignment landmarks
- **plot_run_bouts.py** plots example run-bouts
### ephys_opto
- **all_stim_rasters.py** plots rasters to examine ctrl. versus stim. spiking for all cells
- **nontagged_stim_rasters.py** does a similar thing as the above function but for cells that were not tagged 
- **stim_effect_spikes.py** plots the spike raster and histogram aligned to stimulation pulses for 1 unit as example plots
- **tagged_stim_rasters.py** plots the stimulation-trial rasters for tagged cells
### figure_code
- **plot_all_clustered_acg_heatmap.py** plots the ACGs of each category of LC cells (tagged and putative *Dbh*+ and putative non-*Dbh*+) as heatmaps 
- **plot_avg_acg_tagged.py** plots the mean ACG of tagged *Dbh*+ cells overlaid on single-unit ACG traces
- **plot_demo_tonic_phasic.py** plots artificially-generated examples for tonic versus phasic dynamics
- **plot_eg_tagged_cell_wf.py** plots waveforms of one example tagged cell
- **plot_example_train.py** plots example rasters for one example tagged cell 
- **plot_heatmap_argmax.py** plots superpopulation plot for LC cells, ordered by their run-onset-to-peak latencies
- **plot_heatmap_argmax.py** is similar to the above function but only plots the run-onset peaking cells
- **plot_proportion_bar_graph.py** plots the percentage of tagged cells that are RO-peaking and non-RO-peaking as a bar graph 
- **plot_single_cell_property.py** plots single-cell property (ACG, run-onset-aligned raster, waveform) plots for each recorded unit into \single_cell_property_plots; **plot_single_cell_property_tagged.py** is similar but for tagged cells
- **plot_single_cell_raster_compare.py** plots single-cell rasters aligned to run-onsets, rewards and cues
- **plot_tagged_responses.py** plots single-cell responses to the tagging pulses
- **putative_example_cell.py** plots rasters of example putative *Dbh*+ cells, whereas **tagged_example_cell.py** plots those of example tagged cells 
- **tagged_example_goodvbad_raster.py** plots good versus bad trial rasters for an example cell
### utils 
- **paramC.py** lists parameters (e.g. sampling frequency)
- **RO_peak_detection.py** provides functions for shuffle-based peak-detection
- **single_unit.py** provides basic functions for single-unit analyses

## other_code
currently includes 2 Python scripts to log and plot temperature and humidility recorded from a custom-built ESP8266 circuit; for monitoring lab/2-photon rig temperature and humidity
- run **log_temperature_humidity.py** from Anaconda Prompt after navigating to Dinghao\code_mpfi_dinghao\other_code with `Python log_temperature_humidity.py`