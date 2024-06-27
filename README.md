# code_dinghao_mpfi
my scripts at MPFI

## caiman_code
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
### sequence 
- **HPC_LC_plot_sequence.py** and **HPC_LCterm_plot_sequence.py** plot temporal cell sequences for single sessions
- **HPC_LC_plot_sequence_dist.py** and **HPC_LCterm_plot_sequence_dist.py** plot distance cell sequences for single sessions
### stim_stimcont
### 

## LC_code 
includes analysis scripts for behaviour and neural data collected from locus coeruleus recordings
- **all_earlyvlate_rop_population.py** compares the population spike rates between early 1st-lick trials and late 1st-lick trials; includes tagged, putative and all LC Dbh+ cells
### behaviour 
- **1st_lick_profile.py** plots a stair histogram of the first-lick distance and time of all the trials in LC-optotagging sessions