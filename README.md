# mpfi
my scripts at mpfi

## caiman_code
includes a working CaImAn analysis pipeline modified to work on Wang Lab data, mostly by [Nico Spiller](https://github.com/nspiller).

## dLight_code
includes preliminary analysis scripts for dLight data

## GRABNE_code 
includes preliminary analysis scripts for GRABNE data; currently closely mirrors dLight_code 

## HPC_code
includes analysis scripts for behaviour and neural data collected from hippocampus recordings
essential scripts:
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
- **all_earlyvlate_rop_population.py** compares the population spike rates between early 1st-lick trials and late 1st-lick trials; includes tagged, putative and all LC Dbh+ cells
### behaviour 
- **1st_lick_profile.py** plots a stair histogram of the first-lick distance of all the trials in LC-optotagging sessions