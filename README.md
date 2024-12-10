# code_dinghao_mpfi
my scripts at MPFI; unless otherwise specified, I am the author of all the scripts in this repo

## directory tree
*(does not include defunct scripts)*  
```
├── **HPC_code**  
│   ├── *HPCLC_all_rasters.py* (saves spike raster data for all hippocampal-locus coeruleus (HPC-LC) sessions into .npy files for each session)  
│   ├── *HPCLC_all_train.py* (processes and pools spike train data from all HPC-LC recording sessions, applying Gaussian convolution and saving the results for further analysis)  
│   ├── *HPCLC_clu_list.py* (generates and saves lists of clusters with their metadata (e.g., shank and local cluster ID) for each HPC-LC session as .npy files)  
│   ├── *HPCLC_place_cell_profiles.py* (extracts and saves place cell profiles for each HPC-LC session into a CSV file, including the total number of place cells and their indices)  
│   ├── *HPCLC_pyract_profiles.py* (generates and plots single-cell profiles for pyramidal neurons activated by run-onsets in HPC-LC sessions)  
│   ├── *HPCLC_pyract_single_cell_profiles.py* (creates detailed single-cell profiles for pyramidal neurons in HPC-LC sessions based on activity and Poisson deviations)  
│   ├── *HPCLC_sess_pyr_profiles.py* (plots profiles of all pyramidal neurons for each HPC-LC stimulation session and calculates rise and drop ratios)  
│   ├── *HPCLC_theta_stim.py* (analyses theta rhythm amplitude, frequency, and phase in HPC recordings aligned to LC stimulation)  
│   ├── *HPCLC_waveform_all.py* (summarises the average waveforms of all cells across HPC recordings and saves them in a dictionary)  
│   ├── *HPCLC_waveform_proc.py* (processes and saves average waveforms and standard errors for all clusters in HPC recordings)  
│   ├── ...HPCLCterm versions of the above scripts  
│   ├── *HPC_population_v_licks.py*  
│   ├── *HPC_population_v_licks_poisson.py*  
│   ├── *HPC_population_v_licks_poisson_example_session.py*  
│   ├── *HPC_session_PCA_traj.py*  
│   ├── *HPC_session_UMAP_traj.py*  
│   ├── *HPC_single_trial_UMAP_traj.py*  
│   ├── *all_acg.py*  
│   ├── *all_acg_baseline.py*  
│   ├── **behaviour**  
│   │   ├── *lick_dist_comp_HPC_LC_stim.py*  
│   │   ├── *lick_dist_comp_HPC_LCterm_stim.py*  
│   │   ├── *lick_time_comp_HPC_LC_stim.py*  
│   │   └── *lick_time_comp_HPC_LCterm_stim.py*  
│   ├── **figure_code**  
│   │   ├── *HPCLC_plot_all_rasters.py*  
│   │   ├── *plot_all_pyr_heatmap_dist.py*  
│   │   ├── *plot_all_pyr_heatmap_time_rew.py*  
│   │   ├── *plot_all_pyr_heatmap_time_run.py*  
│   │   ├── *plot_lick_profile_to_pumps_HPCLC.py*  
│   │   └── *plot_lick_profile_to_pumps_HPCLCterm.py*  
│   ├── *lick_history_dependency.py*  
│   ├── *naive_bayesian_decoding_conf_matrix.py*  
│   ├── *naive_bayesian_decoding_proba.py*  
│   ├── *naive_bayesian_decoding_proba_pyract.py*  
│   ├── *naive_bayesian_decoding_proba_pyrinh.py*  
│   ├── *pc_mc_overlaps.py*  
│   ├── *pc_proportion_comp.py*  
│   ├── **sequence**  
│   │   ├── *HPCLC_plot_sequence.py*  
│   │   ├── *HPCLC_plot_sequence_dist.py*  
│   │   ├── *HPCLC_plot_super_sequence.py*  
│   │   ├── *HPCLCterm_plot_sequence.py*  
│   │   └── *HPCLCterm_plot_sequence_dist.py*  
│   ├── **stim_baseline**  
│   │   ├── *HPC_all_stim_baseline_int_only.py*  
│   │   ├── *HPC_all_stim_baseline_pyr_only.py*  
│   │   └── *HPC_all_stim_baseline_pyr_only_PCA.py*  
│   ├── **stim_ctrl**  
│   │   ├── *HPCLC_all_stim_ctrl_int_only.py*  
│   │   ├── *HPCLC_all_stim_ctrl_population_deviation_poisson.py*  
│   │   ├── *HPCLC_all_stim_ctrl_pyr_only.py*  
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
│   │   ├── *all_HPCLC_HPCLCterm_stim_ctrl_int_only_sig.py*  
│   │   ├── *all_HPCLC_HPCLCterm_stim_ctrl_pyr_only_sig.py*  
│   │   └── *all_mod_depth_comp.py*  
│   └── **stim_effect**  
│       ├── *HPCLC_rasters_stim_nonstim.py*  
│       └── *HPC_stim_aligned.py*  
├── **LC_code**  
│   ├── *WT_waveform_proc.py*  
│   ├── *all_UMAP.py*  
│   ├── *all_acg.py*  
│   ├── *all_acg_baseline.py*  
│   ├── *all_cell_properties.py*  
│   ├── *all_earlyvlate_RO_peak.py*  
│   ├── *all_earlyvlate_RO_peak_population.py*  
│   ├── *all_earlyvlate_RO_peak_population_lick2pump.py*  
│   ├── *all_earlyvlate_RO_peak_population_med3licks.py*  
│   ├── *all_earlyvlate_RO_peak_population_raster.py*  
│   ├── *all_earlyvlate_RO_peak_raster.py*  
│   ├── *all_goodvbad_RO_peak.py*  
│   ├── *all_goodvbad_RO_peak_bef.py*  
│   ├── *all_lick_sensitive_activity_1st_lick.py*  
│   ├── *all_lick_sensitive_summary.py*  
│   ├── *all_lick_sensitive_summary_shuf.py*  
│   ├── *all_neu_activity_v_1st_lick.py*  
│   ├── *all_neu_activity_v_1st_lick_shuf.py*  
│   ├── *all_raster_cue_rew_run_lasttocurtr.py*  
│   ├── *all_raster_last_rew_ordered.py*  
│   ├── *all_raster_lick_aligned.py*  
│   ├── *all_raster_lick_ordered.py*  
│   ├── *all_raster_lick_ordered_earlyvlate_only.py*  
│   ├── *all_raster_lick_ordered_raster_only.py*  
│   ├── *all_raster_lick_reward_sensitivity.py*  
│   ├── *all_raster_rew_ordered.py*  
│   ├── *all_raster_rew_to_run_ordered.py*  
│   ├── *all_rasters.py*  
│   ├── *all_rovrb_RO_peak.py*  
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
│   │   ├── *plot_RO_v_nonRO.py*  
│   │   ├── *plot_all_clustered_acg_heatmap.py*  
│   │   ├── *plot_avg_acg_tagged.py*  
│   │   ├── *plot_demo_tonic_phasic.py*  
│   │   ├── *plot_eg_tagged_cell_wf.py*  
│   │   ├── *plot_example_train.py*  
│   │   ├── *plot_heatmap_ROpeaking.py*  
│   │   ├── *plot_heatmap_argmax.py*  
│   │   ├── *plot_proportion_bar_graph.py*  
│   │   ├── *plot_single_cell_ACG.py*  
│   │   ├── *plot_single_cell_property.py*  
│   │   ├── *plot_single_cell_property_tagged.py*  
│   │   ├── *plot_single_cell_raster_RO_aligned.py*  
│   │   ├── *plot_single_cell_waveform.py*  
│   │   ├── *plot_tagged_example_goodvbad_raster.py*  
│   │   ├── *plot_tagged_responses.py*  
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
│   ├── *tagged_rovrb_RO_peak.py*  
│   ├── *tagged_speedvrate.py*  
│   ├── *tagged_train.py*  
│   ├── *tagged_train_alignedRew.py*  
│   ├── *tagged_waveform_classify.py*  
│   ├── *tagged_waveform_pca.py*  
│   ├── *tagging_latency.py*  
│   └── **utils**  
│       ├── *RO_peak_detection.py*  
│       ├── *paramC.py*  
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
│   │   └── *plot_behaviour.py*  
│   ├── *process_behaviour.py*  
│   └── *process_behaviour_imaging.py*  
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
    ├── *behaviour_functions.py*  
    ├── *common.py*  
    ├── *logger_module.py*  
    ├── *param_to_array.py*  
    ├── *plotting_functions.py*  
    ├── *preprocessing.py*  
    └── *read_clu.py*  
```