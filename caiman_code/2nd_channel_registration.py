# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: caiman
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import utils as utl

# %%
# define select tiff files
p_tifs = [*Path(r"C:\temp\A214-20221214-02_data").glob("*tif")]
p_tifs = p_tifs[:3]
p_tifs

# %%
tmp_dir = TemporaryDirectory()
print('created temporary directory', tmp_dir)

# %%

for p_tif in p_tifs:
    utl.split_dual_channel_tif(p_tif, tmp_dir.name)
        

# %%
ch1 = [*Path(tmp_dir.name).glob('*_ch1.tif')]
ch1

# %%
ch2 = [*Path(tmp_dir.name).glob('*_ch2.tif')]
ch2

# %%
import caiman as cm
from caiman.source_extraction.cnmf import params
from caiman.motion_correction import MotionCorrect


# %%
fr = 30  # imaging rate in frames per second
dxy = (1.0, 1.0)  # spatial resolution in x and y in (um per pixel)
# note the lower than usual spatial resolution here
max_shift_um = (12.0, 12.0)  # maximum shift in um
patch_motion_um = (100.0, 100.0)  # patch size for non-rigid correction in um

pw_rigid = True  # flag to select rigid vs pw_rigid motion correction
max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])
overlaps = (24, 24)
max_deviation_rigid = 3

mc_dict = {
    "fnames": ch2,
    "fr": fr,
    "dxy": dxy,
    "pw_rigid": pw_rigid,
    "max_shifts": max_shifts,
    "strides": strides,
    "overlaps": overlaps,
    "max_deviation_rigid": max_deviation_rigid,
    "border_nan": "copy",
}

opts = params.CNMFParams(params_dict=mc_dict)

# %%
c, dview, n_processes = cm.cluster.setup_cluster(
       backend='multiprocessing', n_processes=None, single_thread=False)

# %%
mc = MotionCorrect(ch2, dview=dview, **opts.get_group('motion'))
mc.motion_correct(save_movie=True)

# %%
mmap_file = mc.apply_shifts_movie(ch1, save_memmap=True, save_base_name=f'{tmp_dir.name}/MC', order='C')

# %%
Yr, dims, T = cm.load_memmap(mmap_file)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 

# %%
images.shape

# %%
mmap_file_direct = mc.apply_shifts_movie(ch1, order='F')

# %%
np.array_equal(mmap_file_direct, images)

# %%
tmp_dir.cleanup()
