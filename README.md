# code_dinghao_mpfi
my scripts at MPFI

## caiman_code
*not in use*\
includes a working CaImAn analysis pipeline modified to work on Wang Lab data, mostly by [Nico Spiller](https://github.com/nspiller).

## imaging_code
includes analysis scripts for dLight and GRABNE data
- **run_imaging_pipeline.py** runs grid ROI or Suite2p ROI trace extraction and alignment to behaviour
### defunct_code
- **deepvid_grid_roi.py** is a pipeling to extract fluorescence traces and align them to behavioural variables using a grid ROI mask and denoised movies processed by [Colin Porter](https://github.com/porter-colin93).
- **dLight_all_heatmap.py**
- **dLight_plot.py**
- **GRABNE_grid_roi.py**
- **after_suite2p.py**
- **grid_extract.py**
### suite2p_code
- **registration_roi_extraction.py** registers and extract ROI traces from each recording in lists of sessions listed in rec_list.py using [the Wang lab version of Suite2p](https://github.com/the-wang-lab/suite2p-wang-lab), fine-tuned by [Nico Spiller](https://github.com/nspiller) and [Jingyu Cao](https://github.com/yuxi62).
### tonic_activity 
- **tonic_fft.py** uses FFT to process entire recording traces after calculating dFF (to get rid of the slow decaying of signal strength due to bleaching and evaporation) in order to look for slow periodicity in the signal
### utils
- **imaging_pipeline_functions.py** includes functions used to process imaging recordings
- **imaging_pipeline_main_functions.py** includes grid ROI and Suite2p ROI extraction and alignment functions; these are what run_imaging_pipeline.py calls directly


## HPC_code
includes analysis scripts for behaviour and neural data collected from hippocampus recordings
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
- **all_lick_RO_peak_ordered.py** analyses single cell's response to 1st-licks by ordering the trials based on the time from run-onset to 1st-licks
- **all_lick_sensitive_activity_1st_lick.py** looks at the time from run-onset to neuronal inhibition of cells that are sensitive to 1st-licks; depends on the dataframe output by all_lick_RO_peak_ordered.py
- **all_lick_sensitive.py** also depends on the dataframe output by all_lick_RO_peak_ordered.py and looks at the neuronal inhibition/activation around the 1st-licks; this function edits the dataframe produced by cell_properties.py to add the 'lick_sensitivity' and 'lick_sensitivity_type'
- **all_rasters.py** is a core function that extracts and saves rasters as 0-1 matrices with a structure of trial x time bins
- **all_rasters_cue_rew_run_lasttocurtr.py** does similar things as above, but deals with the spikes before slightly before the cue, the reward and the run-onset of each trial, hence 'lasttocurtr'
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