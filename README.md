# code_dinghao_mpfi
my scripts at MPFI; unless otherwise specified, I am the author of all the scripts in this repo

## directory tree
*(does not include defunct scripts)*  
```
├── **HPC_code**  
│   ├── *HPC_all_extract.py*  
│   ├── *HPC_all_profiles.py*  
│   ├── *HPC_all_waveforms.py*  
│   ├── **bayesian_decoding**  
│   │   ├── *naive_bayesian_decoding_conf_matrix.py*  
│   │   ├── *naive_bayesian_decoding_proba.py*  
│   │   ├── *naive_bayesian_decoding_proba_pyract.py*  
│   │   └── *naive_bayesian_decoding_proba_pyrinh.py*  
│   ├── **behaviour**  
│   │   ├── *lick_dist_comp_HPC_LC_stim.py*  
│   │   ├── *lick_dist_comp_HPC_LCterm_stim.py*  
│   │   ├── *lick_time_comp_HPC_LC_stim.py*  
│   │   └── *lick_time_comp_HPC_LCterm_stim.py*  
│   ├── **dimensionality_reduction**  
│   │   ├── *HPC_UMAP_single_trial_traj_interactive.ipynb*  
│   │   ├── *HPC_single_sess_PCA_traj.py*  
│   │   ├── *HPC_single_sess_UMAP_traj.py*  
│   │   └── *HPC_single_trial_UMAP_traj.py*  
│   ├── **figure_code**  
│   │   ├── *plot_all_ctrl_stim_profiles.py*  
│   │   ├── *plot_all_ctrl_stim_rasters.py*  
│   │   ├── *plot_all_pyr_heatmap_dist.py*  
│   │   ├── *plot_all_pyr_info_ctrl_stim.py*  
│   │   └── *plot_all_pyr_pre_post_ratio.py*  
│   ├── **history_dependency**  
│   │   └── *lick_history_dependency.py*  
│   ├── **lick_sensitivity**  
│   │   ├── *HPCLC_raster_first_lick_ranked.py*  
│   │   └── *HPC_population_activity_1st_lick.py*  
│   ├── **poisson_deviation**  
│   │   └── *HPCLC_pyract_single_cell_profiles.py*  
│   ├── **sequence**  
│   │   ├── *HPCLC_plot_sequence.py*  
│   │   ├── *HPCLC_plot_sequence_dist.py*  
│   │   ├── *HPCLC_plot_super_sequence.py*  
│   │   ├── *HPCLCterm_plot_sequence.py*  
│   │   └── *HPCLCterm_plot_sequence_dist.py*  
│   ├── **stim_ctrl**  
│   │   ├── *HPCLC_all_stim_ctrl_population_deviation_poisson.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_PCA.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_heatmap.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_ordered.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_rasters.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_rasters_blowup.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_rasters_signif_only.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_sig_MI.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only_spatial_info.py*  
│   │   ├── *HPCLC_stim_ctrl_all_int_profiles.py*  
│   │   ├── *HPCLC_stim_ctrl_all_pyr_profiles.py*  
│   │   ├── *HPCLC_stim_ctrl_all_pyr_profiles_single_sess.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_int_only.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only_heatmap.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only_ordered.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only_rasters.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only_rasters_blowup.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only_rasters_signif_only.py*  
│   │   ├── *HPCLCterm_all_stim_ctrl_pyr_only_sig_MI.py*  
│   │   ├── *HPCLCterm_stim_ctrl_all_pyr_profiles_single_sess.py*  
│   │   ├── *HPC_all_mod_depth_comp.py*  
│   │   └── *HPC_all_modulation_statistics.py*  
│   └── **theta_phase**  
│       └── *HPC_all_theta_stim.py*  
├── **LC_code**  
│   ├── *LC_all_profiles.py*  
│   ├── *WT_waveform_proc.py*  
│   ├── *all_UMAP.py*  
│   ├── *all_acg.py*  
│   ├── *all_acg_baseline.py*  
│   ├── *all_earlyvlate_RO_peak.py*  
│   ├── *all_earlyvlate_RO_peak_population.py*  
│   ├── *all_earlyvlate_RO_peak_population_lick2pump.py*  
│   ├── *all_earlyvlate_RO_peak_population_med3licks.py*  
│   ├── *all_earlyvlate_RO_peak_population_raster.py*  
│   ├── *all_earlyvlate_RO_peak_raster.py*  
│   ├── *all_good_bad_RO_peak.py*  
│   ├── *all_goodvbad_RO_peak_bef.py*  
│   ├── *all_neu_activity_ON_OFF.py*  
│   ├── *all_raster_cue_rew_run_lasttocurtr.py*  
│   ├── *all_raster_last_rew_ordered.py*  
│   ├── *all_raster_lick_ordered.py*  
│   ├── *all_raster_lick_ordered_raster_only.py*  
│   ├── *all_raster_lick_reward_sensitivity.py*  
│   ├── *all_raster_rew_ordered.py*  
│   ├── *all_raster_rew_to_run_ordered.py*  
│   ├── *all_rasters.py*  
│   ├── *all_runonset_runbout_RO_peak.py*  
│   ├── *all_time_warped.py*  
│   ├── *all_train.py*  
│   ├── *all_waveform_all.py*  
│   ├── *all_waveform_proc.py*  
│   ├── **behaviour**  
│   │   ├── *1st_lick_profile.py*  
│   │   ├── *egsess_lick.py*  
│   │   ├── *egsess_lick_passive_raphi.py*  
│   │   ├── *egsess_speed.py*  
│   │   ├── *egsess_speed_passive_raphi.py*  
│   │   ├── *good_perc_comp.py*  
│   │   ├── *lick_dist_comp.py*  
│   │   ├── *lick_dist_comp_HPC_LC_stim.py*  
│   │   ├── *lick_history_dependency.py*  
│   │   ├── *lick_history_dependency_comp.py*  
│   │   ├── *lick_properties.py*  
│   │   ├── *lick_time_comp.py*  
│   │   ├── *plot_cue_start_difference.py*  
│   │   ├── *plot_run_bouts.py*  
│   │   ├── *plot_single_trial_example.py*  
│   │   └── *summarise_040.py*  
│   ├── **ephys_opto**  
│   │   ├── *all_stim_rasters.py*  
│   │   ├── *nontagged_stim_rasters.py*  
│   │   ├── *stim_effect_spikes.py*  
│   │   └── *tagged_stim_rasters.py*  
│   ├── **figure_code**  
│   │   ├── *plot_LC_population_heatmap_argmax.py*  
│   │   ├── *plot_all_clustered_acg_heatmap.py*  
│   │   ├── *plot_avg_acg_tagged.py*  
│   │   ├── *plot_eg_tagged_cell_wf.py*  
│   │   ├── *plot_example_train.py*  
│   │   ├── *plot_heatmap_ROpeaking.py*  
│   │   ├── *plot_neu_activity_ON_OFF_mean_profile.py*  
│   │   ├── *plot_proportion_bar_graph.py*  
│   │   ├── *plot_rasters_1st_lick_ordered_early_late_only.py*  
│   │   ├── *plot_rasters_runonset_aligned.py*  
│   │   ├── *plot_runonset_burst_and_non_burst_profiles.py*  
│   │   ├── *plot_single_cell_ACG.py*  
│   │   ├── *plot_single_cell_property.py*  
│   │   ├── *plot_single_cell_property_tagged.py*  
│   │   ├── *plot_single_cell_waveform.py*  
│   │   ├── *plot_tagged_example_good_bad_raster.py*  
│   │   ├── *plot_tagging_responses.py*  
│   │   ├── *plot_trials_LC.py*  
│   │   ├── *putative_example_cell.py*  
│   │   └── *tagged_example_cell.py*  
│   ├── *tag_waveform_all.py*  
│   ├── *tag_waveform_proc.py*  
│   ├── *tag_waveform_proc_notnorm.py*  
│   ├── *tagged_acg_pca.py*  
│   ├── *tagged_goodvbad.py*  
│   ├── *tagged_goodvbad_RO_peak.py*  
│   ├── *tagged_lick_RO_peak_ordered.py*  
│   ├── *tagged_speedvrate.py*  
│   ├── *tagged_train.py*  
│   ├── *tagged_train_alignedRew.py*  
│   ├── *tagged_waveform_classify.py*  
│   ├── *tagged_waveform_pca.py*  
│   ├── *tagging_latency.py*  
│   └── **utils**  
│       └── *single_unit.py*  
├── *README.md*  
├── *Thumbs.db*  
├── **VTA_code**  
│   ├── *all_rasters.py*  
│   ├── *all_train_alignedRew.py*  
│   ├── *all_train_alignedRun.py*  
│   ├── *tag_waveform_proc.py*  
│   ├── *tagged_train_alignedRew.py*  
│   └── *tagged_train_alignedRun.py*  
├── **behaviour_code**  
│   ├── **figure_code**  
│   │   ├── *plot_behaviour.py*  
│   │   └── *plot_lick_to_pumps.py*  
│   ├── *plot_immobile.py*  
│   ├── *process_behaviour.py*  
│   └── *process_behaviour_immobile.py*  
├── **caiman_code**  
│   ├── *2nd_channel_registration.py*  
│   ├── *Untitled.ipynb*  
│   ├── *batch_cnmf.ipynb*  
│   ├── *cnmf.py*  
│   ├── *utils.py*  
│   ├── *utils_mesmerize.py*  
│   └── *visualize.ipynb*  
├── **imaging_code**  
│   ├── *colocalisation_analysis.py*  
│   ├── *convert_movie_tif.py*  
│   ├── *extract_axon_GCaMP.py*  
│   ├── *extract_significant_ROI.py*  
│   ├── **figure_code**  
│   │   ├── *plot_lick_profile.py*  
│   │   ├── *plot_lick_profile_to_pumps.py*  
│   │   ├── *plot_pooled_heatmap_axon_GCaMP.py*  
│   │   └── *plot_std_heatmap.py*  
│   ├── *plot_sorted_heatmaps_grids.py*  
│   ├── *plot_sorted_heatmaps_rois.py*  
│   ├── *run_imaging_pipeline.py*  
│   ├── **suite2p_code**  
│   │   ├── *registration_roi_extraction_s2p_wanglab.py*  
│   │   └── *suite2p-wang-lab_SparseDetect_test_seperate.py*  
│   ├── **tonic_activity**  
│   │   ├── *tonic_fft.py*  
│   │   └── *whole_session_f_dff.py*  
│   ├── **tonic_activity_code**  
│   │   ├── *tonic_fft.py*  
│   │   └── *whole_session_f_dff.py*  
│   └── **utils**  
│       ├── *imaging_pipeline_functions.py*  
│       ├── *imaging_pipeline_main_functions.py*  
│       ├── *imaging_utility_functions.py*  
│       └── *suite2p_functions.py*  
├── **other_code**  
│   ├── *log_temperature_humidity.py*  
│   └── *plot_temperature_humidity.py*  
├── **pharmacology_code**  
│   ├── *summarise_SCH23390.py*  
│   └── *summarise_prop.py*  
└── **utils**  
    ├── *alignment_functions.py*  
    ├── *behaviour_functions.py*  
    ├── *common.py*  
    ├── *dsr1_functions.py*  
    ├── *logger_module.py*  
    ├── *param_to_array.py*  
    ├── *peak_detection_functions.py*  
    ├── *plotting_functions.py*  
    ├── *preprocessing.py*  
    └── *read_clu.py*  
```

## pre-processing
### 2-photon imaging
2-photon imaging data were pre-processed using [suite2p](https://github.com/MouseLand/suite2p)
axon-GCaMP data were pre-processed with a custom-tuned parameter set optimised for identifying neuronal processes
### hippocampus ephys data 
ephys data were spike-sorted using kilosort for pre-processing and manual curation, after which the .res and .clu files were used for trial truncation with a custom MATLAB pipeline. `HPC_all_extract.py` then runs through all hippocampus recordings regardless of whether each recording has been processed, extracting information from the .mat files produced in the previous step; the end results are a smoothed-spike-train file and a raster file for each recording session
`HPC_all_waveforms.py` is used to extract the waveform of each cluster; this is rarely used 
`HPC_all_profiles.py` summarises information on each cluster, including but not limited to `cell_identity` (pyramidal or interneurone), `place_cell` (Boolean), `pre_post` (pre-run-onset/post-run-onset ratio, used to measure the run-onset response), `SI` (spatial info.), `TI` (temporal info.), `prof_mean` (spike rate profile mean)