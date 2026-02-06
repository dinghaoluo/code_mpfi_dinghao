# code_dinghao_mpfi
my scripts at MPFI; unless otherwise specified, I am the author of all the scripts in this repo

## setup
make sure that `...\code_mpfi_dinghao\utils` is in system paths  
in addition, the recording list (`rec_list`) is stored in the lab network drive (`Z:\Dinghao\code_dinghao`)
to add both to system paths:
- Windows: in PowerShell, enter `setx PYTHONPATH "Z:\Dinghao\code_mpfi_dinghao\utils;Z:\Dinghao\code_dinghao"` (repeated `setx`'s overwrite previous paths)
- Linux/macOS: launch Terminal, open the shell config file with `nano ~/.zshrc`; at the bottom of the config file add `export PYTHONPATH="/mnt/z/Dinghao/code_mpfi_dinghao/utils:/mnt/z/Dinghao/code_dinghao:$PYTHONPATH"`

## directory tree
*(does not include defunct scripts)*  
```
├── **HPC_code**  
│   ├── *HPC_all_extract.py*  
│   ├── *HPC_all_extract_raphi.py*  
│   ├── *HPC_all_profiles.py*  
│   ├── *HPC_all_profiles_raphi.py*  
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
│   ├── **crossover_point**  
│   │   └── *crossover_point_analysis.py*  
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
│   │   ├── *plot_all_pyr_pre_post_ratio.py*  
│   │   ├── *plot_all_pyr_pre_post_raw_change.py*  
│   │   ├── *plot_run_onset_ON_OFF_profiles.py*  
│   │   └── *plot_run_onset_ON_OFF_profiles_raphi.py*  
│   ├── **first_lick_analysis**  
│   │   ├── *all_earlyvlate_pyr_fixed_threshold_mean_std.py*  
│   │   ├── *all_earlyvlate_pyr_fixed_threshold_mean_std_raphi.py*  
│   │   ├── *all_earlyvlate_pyr_fixed_threshold_mean_std_raphi_thres15.py*  
│   │   └── *all_earlyvlate_speed_fixed_threshold.py*  
│   ├── **lick_sensitivity**  
│   │   ├── *HPCLC_raster_first_lick_ranked.py*  
│   │   ├── *HPC_early_late_first_lick_profiles.py*  
│   │   ├── *HPC_early_late_first_lick_proportions.py*  
│   │   └── *HPC_population_activity_1st_lick.py*  
│   ├── **poisson_deviation**  
│   │   └── *HPCLC_pyract_single_cell_profiles.py*  
│   ├── **remapping**  
│   │   └── *HPC_global_remapping_pop_vector.py*  
│   ├── **sequence**  
│   │   ├── *HPCLC_plot_sequence.py*  
│   │   ├── *HPCLC_plot_sequence_dist.py*  
│   │   ├── *HPCLC_plot_super_sequence.py*  
│   │   ├── *HPCLCterm_plot_sequence.py*  
│   │   └── *HPCLCterm_plot_sequence_dist.py*  
│   ├── **stim_ctrl**  
│   │   ├── *all_stim_ctrl_effects.py*  
│   │   └── *all_stim_ctrl_pyr_ON_OFF.py*  
│   └── **theta_phase**  
│       └── *HPC_all_theta_stim.py*  
├── **IBL_code**  
│   └── *test.py*  
├── **LC_code**  
│   ├── **GLM**  
│   │   ├── *GLM_LC_beh_permutation.py*  
│   │   ├── *GLM_LC_beh_permutation_full.py*  
│   │   ├── *amp_autocorrelagram.py*  
│   │   ├── *amp_baseline_rate.py*  
│   │   ├── *amp_since_last_reward.py*  
│   │   ├── *amp_since_last_reward_binned.py*  
│   │   └── *tonic_fft_LC.py*  
│   ├── *LC_all_extract_all.py*  
│   ├── *LC_all_identity_UMAP.py*  
│   ├── *LC_all_profiles.py*  
│   ├── *LC_all_spikes_ISIs.py*  
│   ├── *LC_all_waveforms_acgs.py*  
│   ├── *LC_run_all.py*  
│   ├── **alignment_analysis**  
│   │   └── *analyse_alignment_with_heatmap_run_cue_rew_aligned.py*  
│   ├── **behaviour**  
│   │   ├── *1st_lick_profile.py*  
│   │   ├── *ctrl_stim_lick_properties.py*  
│   │   ├── *egsess_lick.py*  
│   │   ├── *egsess_lick_passive_raphi.py*  
│   │   ├── *egsess_speed.py*  
│   │   ├── *egsess_speed_passive_raphi.py*  
│   │   ├── *good_perc_comp.py*  
│   │   ├── *lick_dist_comp_020.py*  
│   │   ├── *lick_dist_comp_HPC_LC_stim.py*  
│   │   ├── *lick_history_dependency.py*  
│   │   ├── *lick_history_dependency_comp.py*  
│   │   ├── *lick_time_comp_020.py*  
│   │   ├── *plot_cue_start_difference.py*  
│   │   ├── *plot_run_bouts.py*  
│   │   ├── *plot_run_bouts_examples.py*  
│   │   └── *plot_single_trial_example.py*  
│   ├── **ephys_opto**  
│   │   └── *analyse_stim_response.py*  
│   ├── **figure_code**  
│   │   ├── *plot_ISIs.py*  
│   │   ├── *plot_acgs_and_heatmap.py*  
│   │   ├── *plot_comp_tagged_putative.py*  
│   │   ├── *plot_ctrl_stim_profiles.py*  
│   │   ├── *plot_neu_activity_ON_OFF_mean_profile.py*  
│   │   ├── *plot_rasters_1st_lick_ordered_early_late_only.py*  
│   │   ├── *plot_rasters_run_cue_rew_aligned.py*  
│   │   ├── *plot_runonset_burst_and_non_burst_profiles.py*  
│   │   ├── *plot_single_cell_ACG.py*  
│   │   ├── *plot_single_cell_waveform.py*  
│   │   ├── *plot_tagged_example_good_bad_raster.py*  
│   │   ├── *plot_tagging_responses.py*  
│   │   └── *plot_trials_LC.py*  
│   ├── **first_lick_analysis**  
│   │   ├── *all_earlyvlate_RO_peak_fixed_threshold.py*  
│   │   ├── *all_neu_activity_ON_OFF.py*  
│   │   └── *all_time_warped.py*  
│   ├── **good_v_bad_trials**  
│   │   ├── *all_good_bad_RO_peak.py*  
│   │   └── *all_goodvbad_RO_peak_bef.py*  
│   ├── **rasters**  
│   │   ├── *all_raster_cue_rew_run_lasttocurtr.py*  
│   │   ├── *all_raster_last_rew_ordered.py*  
│   │   ├── *all_raster_lick_ordered.py*  
│   │   ├── *all_raster_lick_ordered_raster_only.py*  
│   │   ├── *all_raster_lick_reward_sensitivity.py*  
│   │   ├── *all_raster_rew_ordered.py*  
│   │   └── *all_raster_rew_to_run_ordered.py*  
│   ├── **run_onset_burst_analysis**  
│   │   ├── *burst_detection.py*  
│   │   └── *early_v_late_burst_probability.py*  
│   ├── **run_onset_v_run_bout**  
│   │   └── *all_runonset_runbout_RO_peak.py*  
│   └── **tagging_analysis**  
│       └── *tagging_latency.py*  
├── *README.md*  
├── *Thumbs.db*  
├── **VTA_code**  
│   ├── *all_rasters.py*  
│   ├── *all_train_alignedRew.py*  
│   ├── *all_train_alignedRun.py*  
│   ├── *tag_waveform_proc.py*  
│   ├── *tagged_train_alignedRew.py*  
│   └── *tagged_train_alignedRun.py*  
├── **_supp_figures_external**  
│   ├── *FigureSupp1.pdf*  
│   ├── *FigureSupp2.pdf*  
│   ├── *FigureSupp3.pdf*  
│   └── *FigureSupp6.pdf*  
├── **behaviour_code**  
│   ├── *analyse_pupil_size.py*  
│   ├── *analyse_speed_licks.py*  
│   ├── **behaviour_control**  
│   │   ├── *HPC_opto_speed_controls.py*  
│   │   ├── *LC_controls.py*  
│   │   └── *LC_opto_speed_controls.py*  
│   ├── **figure_code**  
│   │   ├── *plot_example_session.py*  
│   │   ├── *plot_example_trials.py*  
│   │   ├── *plot_immobile.py*  
│   │   ├── *plot_speeds.py*  
│   │   └── *plot_trial_by_trial.py*  
│   ├── *first_lick_since_last_reward.py*  
│   ├── *off_target_run_bouts.py*  
│   ├── **optogenetics**  
│   │   └── *summarise_opto.py*  
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
├── **history_dependency_code**  
│   └── *lick_history_dependency.py*  
├── **imaging_code**  
│   ├── *HPC_GRABNE_LC_opto_extract.py*  
│   ├── *HPC_GRABNE_tone_extract.py*  
│   ├── *HPC_dLight_LC_opto_extract.py*  
│   ├── *HPC_extract_significant_ROI.py*  
│   ├── *HPC_run_imaging_pipeline.py*  
│   ├── *LCHPC_axon_all_extract.py*  
│   ├── *LCHPC_axon_all_extract_immobile.py*  
│   ├── *LCHPC_axon_all_profiles.py*  
│   ├── *LCHPC_single_pixel_extract.py*  
│   ├── **ROI_vs_neuropil**  
│   │   ├── *ROI_vs_neuropil_RI_mean.py*  
│   │   └── *ROI_vs_neuropil_RI_over_time.py*  
│   ├── *Suite2p_registration.py*  
│   ├── **alignment_analysis**  
│   │   └── *analyse_alignment_with_heatmap_run_cue_rew_aligned.py*  
│   ├── **controls**  
│   │   └── *dLight_expression_control.py*  
│   ├── *convert_movie_tif_GUI.py*  
│   ├── **dLight_inhibition**  
│   │   └── *HPC_dLight_LC_inh_stim_ctrl_run.py*  
│   ├── **dLight_stim_dispersion**  
│   │   ├── *single_ROI_binned_dilation.py*  
│   │   ├── *single_ROI_binned_dilation_spatial_tau.py*  
│   │   ├── *single_ROI_binned_dispersion.py*  
│   │   └── *whole_field_binned_dispersion.py*  
│   ├── **fibre_segger_GUI**  
│   │   ├── *fibre-segmenter.ico*  
│   │   ├── *fibre_ROI_segmentation.py*  
│   │   ├── *fibre_ROI_segmentation_GUI_v1.py*  
│   │   ├── *fibre_ROI_segmentation_GUI_v2.py*  
│   │   ├── *fibre_ROI_segmentation_GUI_v3.py*  
│   │   └── *fibre_ROI_segmentation_GUI_v4.py*  
│   ├── **figure_code**  
│   │   ├── *example_sess_refs_release_tiff.py*  
│   │   ├── *plot_16_bit_maps.py*  
│   │   ├── *plot_dLight_LC_opto_single_axon_stim_profiles.py*  
│   │   ├── *plot_lick_profile.py*  
│   │   ├── *plot_lick_profile_to_pumps.py*  
│   │   ├── *plot_pooled_heatmap_axon_GCaMP.py*  
│   │   ├── *plot_raw traces_axon_GCaMP.py*  
│   │   ├── *plot_raw traces_axon_GCaMP_example_trials.py*  
│   │   ├── *plot_sorted_heatmaps_grids.py*  
│   │   ├── *plot_sorted_heatmaps_rois.py*  
│   │   ├── *plot_std_heatmap.py*  
│   │   ├── *plot_whole_field.py*  
│   │   ├── *summarise_dLight_LC_opto_all.py*  
│   │   └── *summarise_dLight_LC_opto_ctrl_inh.py*  
│   ├── **first_lick**  
│   │   └── *LCaxon_earlyvlate_RO_peak_fixed_threshold.py*  
│   ├── **optogenetics**  
│   │   ├── *dLight_LC_opto_release_stim_ctrl.py*  
│   │   └── *summarise_opto_imaging.py*  
│   ├── **release_probability**  
│   │   ├── *prop_signif_release_dLight_stim.py*  
│   │   └── *release_probability_dLight_stim.py*  
│   ├── **suite2p_code**  
│   │   ├── *registration_roi_extraction_s2p_wanglab.py*  
│   │   └── *suite2p-wang-lab_SparseDetect_test_seperate.py*  
│   ├── **test_whole_field_pipeline**  
│   │   └── *test_whole_field_pipeline.py*  
│   └── **tonic_activity**  
│       ├── *tonic_fft.py*  
│       └── *whole_session_f_dff.py*  
├── **matlab_preprocessing**  
│   ├── *RunSpikePipeline.m*  
│   ├── *RunSpikePipeline_pix.m*  
│   └── *RunSpikePipeline_pix_Run0.m*  
├── **modelling_code**  
│   └── *general_model.py*  
├── **other_code**  
│   ├── *log_temperature_humidity.py*  
│   ├── *plot_model_schematic.py*  
│   └── *plot_temperature_humidity.py*  
├── **pharmacology_code**  
│   ├── *summarise_SCH23390.py*  
│   ├── *summarise_prazosin.py*  
│   └── *summarise_propranolol.py*  
└── **utils**  
    ├── *GLM_functions.py*  
    ├── *alignment_functions.py*  
    ├── *behaviour_functions.py*  
    ├── *common_functions.py*  
    ├── *decay_time_analysis.py*  
    ├── *dsr1_functions.py*  
    ├── *history_dependency_functions.py*  
    ├── *imaging_pipeline_functions.py*  
    ├── *imaging_pipeline_main_functions.py*  
    ├── *imaging_utility_functions.py*  
    ├── *logger_module.py*  
    ├── *param_to_array.py*  
    ├── *peak_detection_functions.py*  
    ├── *plotting_functions.py*  
    ├── *preprocessing.py*  
    ├── *read_clu.py*  
    ├── *single_unit.py*  
    ├── *suite2p_functions.py*  
    ├── *support_HPC.py*  
    └── *support_LCHPC_axon.py*  
``````

## pre-processing

### behavior analysis 

Behaviour analysis of the running VR task is handled by scripts under `/behaviour_code`. `process_behaviour.py` process all sessions from all experiments, saving the processed behavior data as a `.pkl` file under each session's data folder. This `.pkl` file can be straightforwardly loaded elsewhere with `pickle.load`.

Behaviour analysis of the immobile VR task is handled, alternatively, by `process_behaviour_immobile.py`. 

`off_target_run_bouts.py` is a Python implementation of the run-bout detection algorithm in the MATLAB pipeline and `plot_run_bouts.py` in `~/figure_code` plots the spike-rate profiles of (currently only) locus coeruleus cells on top of velocity curves of animals, with licks, run-onsets and run bout-onsets identified.

### 2-photon imaging

2-photon imaging data were pre-processed using [suite2p](https://github.com/MouseLand/suite2p)  

**axon-GCaMP recordings**: after sorting using Suite2p with customised parameters to detect neuronal processes, `extract_axon_GCaMP.py` extracts dF/F traces aligned to behavioural landmarks (e.g. run-onsets, reward deliveries) of valid ROIs. Suite2p saves sorted ROIs in such a manner that each ROI has an `imerge` list consisting of all of its constituent ROIs and ROIs resulting from multiple merges would contain an `imerge` list that is a superset of all of the constituents of ROIs from previous merging steps. Therefore, a `valid_ROI_dict` is generated, containing only the ROIs from the final merge step (i.e. which are not themselves constituents of other ROIs).  

**neuromodulator sensor recordings**: after registration (and ROI detection) using Suite2p, `run_imaging_pipeline.py` provides 2 ways to process the data based on grid-like ROIs and Suite2p ROIs. Grid-like ROIs divide the imaging plane into square grids and extract traces based strictly within those grids, without spatial filtering. Suite2p ROIs are detected using a customised parameter set to prioritise temporal variances of detected ROIs.

**opto-neuromodulator sensor recordings**: when optogenetics is performed simultaneously with imaging, one can use `HPC_dLight_LC_opto_extract.py` (named so since currently only HPC-dLight + LC-opto recordings are being performed) to extract a) aligned-to-stim. activity traces and b) single-pixel aligned traces for future use. The 'future use' for now consists of using `HPC_dLight_LC_opto_pixel_wise_map` to extract a pixel-wise t-map to ascertain the areas of highest and most consistent changes in response to stimulations.

### hippocampus ephys data 

ephys data were spike-sorted using kilosort for pre-processing and manual curation, after which the .res and .clu files were used for trial truncation with a custom MATLAB pipeline. `HPC_all_extract.py` then runs through all hippocampus recordings regardless of whether each recording has been processed, extracting information from the .mat files produced in the previous step; the end results are a smoothed-spike-train file and a raster file for each recording session  
`HPC_all_waveforms.py` is used to extract the waveform of each cluster; this is rarely used  
`HPC_all_profiles.py` summarises information on each cluster, including but not limited to `cell_identity` (pyramidal or interneurone), `place_cell` (Boolean), `pre_post` (pre-run-onset/post-run-onset ratio, used to measure the run-onset response), `SI` (spatial info.), `TI` (temporal info.), `prof_mean` (spike rate profile mean)