# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 18:51:23 2022

@author: Dinghao Luo
"""

import numpy as np

def param2array(filename):  # convert the param files to numpy arrays

    with open(filename, "r", encoding='utf-8-sig') as f:
        string = f.read()

    prsd = string.split('\n')  # parse parameter file using \n as delimiter
    v = np.asarray(prsd)

    return v


def get_clu(n, clu):  # get spike ID for a given clu
    
    if type(n) == int:
        n = str(n)
    
    clu_id = np.array(np.where(clu == n))
    
    return clu_id


def get_clu_tp(n, res, clu):  # get spike time points for a given clu
    
    if type(n) == int:
        n = str(n)
    
    clu_t = np.array(res[np.where(clu == n)])
    # clu_t = [int(numeric_string)/20000 for numeric_string in clu_t]  # convert to ms
    
    return clu_t