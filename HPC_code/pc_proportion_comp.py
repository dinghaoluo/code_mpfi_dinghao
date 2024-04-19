# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:50:42 2024

compare the proportion of place cells in regulated populations

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys 
import pandas as pd
from scipy.stats import linregress

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% load paths and data 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
pathHPC = rec_list.pathHPCLCopt
df = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv') 
pathHPCterm = rec_list.pathHPCLCtermopt
dfterm = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.csv')


#%% containers 
pc_props = []  # proportions of PCs
mc_props = []  # proportions of modulated cells
uc_props = []  # proportions of mod-up cells
dc_props = []  # proportions of mod-down cells 


#%% main loop 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    tot_pc = len(place_cells)
    
    if tot_pc!=0:
            
        # initialise local variables
        mc = 0  # modulated 
        uc = 0  # mod-up
        dc = 0  # mod-down 
            
        clu_list = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_clu_list_{}.npy'.format(recname, recname))
        tot_clu = len(clu_list)
        
        # if each neurones is an interneurone or a pyramidal cell 
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        rec_info = info['rec'][0][0]
        intern_id = rec_info['isIntern'][0]
        pyr_id = [not(clu) for clu in intern_id]
        tot_clu = len(pyr_id)
        
        # loop over all
        for i in range(tot_clu):
            if pyr_id[i]==True:
                cluname = clu_list[i]
                reg = df.loc[df['Unnamed: 0'] == cluname]['regulated'].item()
                if reg!='none':
                    mc+=1
                    if reg=='up':
                        uc+=1
                    else:
                        dc+=1
        
        if mc>0 and uc>0 and dc>0:       
            pc_props.append(np.log(tot_pc/tot_clu))
            mc_props.append(np.log(mc/tot_clu))
            uc_props.append(np.log(uc/tot_clu))
            dc_props.append(np.log(dc/tot_clu))

for pathname in pathHPCterm:
    recname = pathname[-17:]
    print(recname)
    
    classification = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_FieldSpCorrAligned_Run1_Run0.mat'.format(pathname, recname))
    place_cells = classification['fieldSpCorrSessNonStimGood'][0][0]['indNeuron'][0]
    tot_pc = len(place_cells)
    
    if tot_pc!=0:
        
        # initialise local variables
        mc = 0  # modulated 
        uc = 0  # mod-up
        dc = 0  # mod-down 
            
        clu_list = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_clu_list_{}.npy'.format(recname, recname))
        tot_clu = len(clu_list)
        
        # if each neurones is an interneurone or a pyramidal cell 
        info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
        rec_info = info['rec'][0][0]
        intern_id = rec_info['isIntern'][0]
        pyr_id = [not(clu) for clu in intern_id]
        tot_clu = len(pyr_id)
        
        # loop over all
        for i in range(tot_clu):
            if pyr_id[i]==True:
                cluname = clu_list[i]
                reg = dfterm.loc[dfterm['Unnamed: 0'] == cluname]['regulated'].item()
                if reg!='none':
                    mc+=1
                    if reg=='up':
                        uc+=1
                    else:
                        dc+=1
        
        if mc>0 and uc>0 and dc>0:  
            pc_props.append(np.log(tot_pc/tot_clu))
            mc_props.append(np.log(mc/tot_clu))
            uc_props.append(np.log(uc/tot_clu))
            dc_props.append(np.log(dc/tot_clu))


#%% as arrays 
pc_props = np.array(pc_props)
mc_props = np.array(mc_props)
uc_props = np.array(uc_props)
dc_props = np.array(dc_props)


#%% statistics
slope, interc, rval, pval, stderr = linregress(pc_props, mc_props)

    
#%% plotting 
fig, ax = plt.subplots(figsize=(3,3))

xaxis = np.linspace(-6, 0)

ax.scatter(pc_props, mc_props, c='k', ec='none', s=5)
ax.plot(xaxis, slope*xaxis+interc, c='darkblue')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.set(xlabel='log PC prop.', ylabel='log mod. prop.',
       xlim=(-5.5, -.5), ylim=(-4, -.5),
       xticks=[-5,-3,-1], yticks=[-1, -3],
       title='stderr={}\npval={}'.format(stderr, pval))

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\pc_mc_proportion_comp.png',
            dpi=300,
            bbox_inches='tight')


#%% statistics up
slope, interc, rval, pval, stderr = linregress(pc_props, uc_props)

    
#%% plotting 
fig, ax = plt.subplots(figsize=(3,3))

xaxis = np.linspace(-6, 0)

ax.scatter(pc_props, uc_props, c='darkorange', ec='none', s=5)
ax.plot(xaxis, slope*xaxis+interc, c='darkblue')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.set(xlabel='log PC prop.', ylabel='log up_mod. prop.',
       xlim=(-5.5, -.5), ylim=(-4, -.5),
       xticks=[-5,-3,-1], yticks=[-1, -3],
       title='stderr={}\npval={}'.format(stderr, pval))

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\pc_uc_proportion_comp.png',
            dpi=300,
            bbox_inches='tight')


#%% statistics down
slope, interc, rval, pval, stderr = linregress(pc_props, dc_props)


#%% plotting 
fig, ax = plt.subplots(figsize=(3,3))

xaxis = np.linspace(-6, 0)

ax.scatter(pc_props, dc_props, c='darkgreen', ec='none', s=5)
ax.plot(xaxis, slope*xaxis+interc, c='darkblue')

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

ax.set(xlabel='log PC prop.', ylabel='log down-mod. prop.',
       xlim=(-5.5, -.5), ylim=(-4, -.5),
       xticks=[-5,-3,-1], yticks=[-1, -3],
       title='stderr={}\npval={}'.format(stderr, pval))

fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\pc_dc_proportion_comp.png',
            dpi=300,
            bbox_inches='tight')