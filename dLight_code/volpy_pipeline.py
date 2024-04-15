# -*- coding: utf-8 -*-
"""
Created on Thu 21 Dec 13:44:34 2023

VolPy implementation on dLight data

@author: Dinghao Luo
"""

#%% imports 
from base64 import b64encode
import cv2
import glob
import h5py
import imageio
from IPython import get_ipython
from IPython.display import HTML, display, clear_output
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        get_ipython().run_line_magic('matplotlib', 'qt')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.source_extraction.volpy.mrcnn import visualize, neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.paths import caiman_datadir

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)


#%%