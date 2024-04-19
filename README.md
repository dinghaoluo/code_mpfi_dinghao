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
- HPCLC_all_rasters.py and HPCLCterm_all_rasters.py read spiketime data from behaviour-aligned spiketime files and produce exactly one 1-0 raster array for every recording session
- HPCLC_all_train.py and HPCLCterm_all_train.py read spiketime data similarly to ...rasters.py, but then convolve the spike train with a 100-ms-sigma Gaussian kernel and produce exactly one spike train array for every recording session
- HPCLC_clu_list.py and HPCLCterm_clu_list.py generate a list of clu name for every recording session to accelerate later processing
- HPCLC_place_cell_profiles.py and HPCLCterm_place_cell_profiles.py summarise the place cell indices and number of place cells for each recording session
### stim_stimcont
