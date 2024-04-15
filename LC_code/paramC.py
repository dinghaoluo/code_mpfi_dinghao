# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:12:42 2023

paramC

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 


#%% Gaussian kernel for spike convolution
x = np.arange(-500, 500, 1)
sigma = 125

gaus_spike = [1 / (sigma*np.sqrt(2*np.pi)) * 
              np.exp(-t**2/(2*sigma**2)) for t in x]