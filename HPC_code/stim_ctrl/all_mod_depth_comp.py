# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:53:22 2024

check the depths of regulated cells 

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import sys 
import pandas as pd
from scipy.stats import ranksums

# plotting parameters 
import matplotlib
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% switches 
# paths to load, 1 = HPCLC, 0 = HPCLCterm
HPC_LC = 0


#%% load paths and data 
if ('Z:\Dinghao\code_dinghao' in sys.path) == False:
    sys.path.append('Z:\Dinghao\code_dinghao')
import rec_list
if HPC_LC:
    pathHPC = rec_list.pathHPCLCopt
    df = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_diff_profiles_pyr_only.csv') 
elif not HPC_LC:
    pathHPC = rec_list.pathHPCLCtermopt
    df = pd.read_csv('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_diff_profiles_pyr_only.csv')


#%% containers 
reg_depth = []
non_depth = []
rup_depth = []  # up-regulated 
rdw_depth = []  # dw-regulated


#%% main loop 
for pathname in pathHPC:
    recname = pathname[-17:]
    print(recname)
    
    # load trains for this recording 
    all_info = np.load('Z:\Dinghao\code_dinghao\HPC_all\{}\HPC_all_info_{}.npy'.format(recname, recname),
                       allow_pickle=True).item()
    
    tot_time = 5 * 1250  # 5 seconds in 1250 Hz
    
    count_sensitive = 0  # how many cells in this session respond to stims
    count_exc = 0
    count_inh = 0
    
    trains = list(all_info.values())
    clu_list = list(all_info.keys())
    tot_trial = len(trains[0])
    
    # if each neurones is an interneurone or a pyramidal cell 
    info = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Info.mat'.format(pathname, recname))
    rec_info = info['rec'][0][0]
    intern_id = rec_info['isIntern'][0]
    pyr_id = [not(clu) for clu in intern_id]
    tot_clu = len(pyr_id)

    # depth 
    depth = sio.loadmat('{}\{}_DataStructure_mazeSection1_TrialType1_Depth.mat'.format(pathname, recname))['depthNeu'][0]
    rel_depth = depth['relDepthNeu'][0][0]
    
    # loop over all
    for i in range(tot_clu):
        if pyr_id[i]==True:
            cluname = clu_list[i]
            d = rel_depth[i]
            reg = df.loc[df['Unnamed: 0'] == cluname]['regulated'].item()
            if reg=='none':
                non_depth.append(d)
            else:
                reg_depth.append(d)
                if reg=='up':
                    rup_depth.append(d)
                else:
                    rdw_depth.append(d)
                
    
#%% statistics 
"""
Note that here using ranksums (Mann-Whitney U) is justified, since we are 
concerned about the distribution of the depths, and the data we have are not
paired (signed-rank is not applicable). In addition, this means that we would 
prefer to show the data with violin plots.
"""
non_v_reg = ranksums(non_depth, reg_depth)[1]
rup_v_rdw = ranksums(rup_depth, rdw_depth)[1]

    
#%% plotting 
fig, ax = plt.subplots(figsize=(3,4))

vp = ax.violinplot([non_depth, reg_depth],
                   showmeans=True, showextrema=False)
vp['bodies'][0].set_color('grey')
vp['bodies'][1].set_color('k')
for i in [0,1]:
    vp['bodies'][i].set_edgecolor('none')
vp['cmeans'].set(color='darkred')

jit_non = np.random.uniform(-.03, .03, len(non_depth))
jit_non_y = np.random.uniform(-.5, .5, len(non_depth))
jit_reg = np.random.uniform(-.03, .03, len(reg_depth))
jit_reg_y = np.random.uniform(-.5, .5, len(reg_depth))
ax.scatter([1]*len(non_depth)+jit_non, non_depth+jit_non_y, s=3, c='grey', ec='none', alpha=.3)
ax.scatter([2]*len(reg_depth)+jit_reg, reg_depth+jit_reg_y, s=3, c='k', ec='none', alpha=.3)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(.5,2.5), ylim=(-7.1,8.8),
       xticks=[1,2], xticklabels=['non-mod.', 'mod.'],
       ylabel='depth rel. S. pyramidale',
       title='ranksums p={}'.format(np.round(non_v_reg,8)))

fig.tight_layout()
plt.show()

if HPC_LC==1:
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_depths_non_v_reg.png',
                dpi=300,
                bbox_inches='tight')
else:
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_depths_non_v_reg.png',
                dpi=300,
                bbox_inches='tight')


#%% plotting (up v dw)
fig, ax = plt.subplots(figsize=(3, 4))

vps = ax.violinplot([rup_depth, rdw_depth],
                    showmeans=True, showextrema=False)
vps['bodies'][0].set_color('darkorange')
vps['bodies'][1].set_color('darkgreen')
for i in [0,1]:
    vps['bodies'][i].set_edgecolor('none')
vps['cmeans'].set(color='darkred')

jit_rup = np.random.uniform(-.03, .03, len(rup_depth))
jit_rup_y = np.random.uniform(-.5, .5, len(rup_depth))
jit_rdw = np.random.uniform(-.03, .03, len(rdw_depth))
jit_rdw_y = np.random.uniform(-.5, .5, len(rdw_depth))
ax.scatter([1]*len(rup_depth)+jit_rup, rup_depth+jit_rup_y, s=3, c='darkorange', ec='none', alpha=.3)
ax.scatter([2]*len(rdw_depth)+jit_rdw, rdw_depth+jit_rdw_y, s=3, c='darkgreen', ec='none', alpha=.3)

for s in ['top', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set(xlim=(.5,2.5), ylim=(-7.1,8.8),
       xticks=[1,2], xticklabels=['up-mod.', 'down-mod.'],
       ylabel='depth rel. S. pyramidale',
       title='ranksums p={}'.format(np.round(rup_v_rdw,5)))

fig.tight_layout()
plt.show()

if HPC_LC==1:
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LC_stim_stimcont_depths_rup_v_rdw.png',
                dpi=300,
                bbox_inches='tight')
else:
    fig.savefig('Z:\Dinghao\code_dinghao\HPC_all\HPC_LCterm_stim_stimcont_depths_rup_v_rdw.png',
                dpi=300,
                bbox_inches='tight')