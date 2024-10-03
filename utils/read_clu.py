#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
a simple script to read and visualise spike train data from .clu and .res files
Created on Sat Oct 16 18:08:41 2021
@author: dinghaoluo
"""

#--------imports--------#

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#-------functions-------#

def param2array(filename): #convert the param files to numpy arrays
    
    with open(filename, "r", encoding='utf-8-sig') as f:
        string = f.read()
    
    prsd = string.split('\n') #parse parameter file using \n as delimiter
    
    v = np.asarray(prsd)
    
    return v #return 1d array


def get_clu(n, res, clu):  # get spike time points for a given clu
    
    if type(n) == int:
        n = str(n)
    
    clu_t = np.array(res[np.where(clu == n)])
    clu_t = [int(numeric_string)/20000 for numeric_string in clu_t]
    
    return clu_t


def get_clu_20kHz(n, res, clu):  # get spike time points for a given clu
    
    if type(n) == int:
        n = str(n)
    
    clu_t = np.array(res[np.where(clu == n)])
    clu_t = [int(numeric_string) for numeric_string in clu_t]
    
    return clu_t


def get_fr(clu, res):
    
    n = np.size(clu)
    T = int(res[-2])/20000
    
    fr = n / T
    
    return fr


def frfilt_low_pass(clus_dict, res_file, n):
    
    clus_dict_filtered = {}
    
    clus_dict_filtered['1'] = clus_dict['1'] #multi-unit cluster
    
    for i in clus_dict.keys():
        if get_fr(clus_dict[i], res_file) < n:
            clus_dict_filtered[i] = clus_dict[i]
        
    return clus_dict_filtered