# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:22:04 2024

plotting functions to save us from the chaos of having to code out the plotting
    section again and again without salvation which has been extremely painful
    and i do not understand why i did not do this earlier

@author: Dinghao Luo
"""


#%% imports 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import wilcoxon, ranksums, ttest_rel, ttest_ind


#%% functions 
def plot_violin_with_scatter(data1, data2, colour1, colour2,
                             paired=True,
                             xlabel=' ', xticks=[1,2], xticklabels=['data1', 'data2'],
                             ylabel=' ', yscale=None,
                             title=' ',
                             showmeans=False, showmedians=True, showextrema=False,
                             statistics=True,
                             save=False, savepath=' ', dpi=120):
    """
    pretty self-explanatory; plots half-violins on x-positions 1 and 2
    useful for comparing ctrl and stim conditions
    if paired==True, use paired statistics 
    if paired==False, use unpaired statistics 
    """
    fig, ax = plt.subplots(figsize=(1.8,2.4))
    
    vp = ax.violinplot([data1, data2],
                       positions=[1,2],
                       showmeans=showmeans, showmedians=showmedians, 
                       showextrema=False)

    vp['bodies'][0].set_color(colour1)
    vp['bodies'][1].set_color(colour2)
    if showmeans:
        vp['cmeans'].set_color('k')
        vp['cmeans'].set_linewidth(2)
        ax.scatter(1.1, np.mean(data1), 
                   s=30, c=colour1, ec='none', lw=.5, zorder=2)
        ax.scatter(1.9, np.mean(data2), 
                   s=30, c=colour2, ec='none', lw=.5, zorder=2)
        if paired:
            ax.plot([1.1, 1.9], 
                    [data1, data2], 
                    color='grey', alpha=.05, linewidth=1, zorder=1)
            ax.plot([1.1, 1.9], [np.mean(data1), np.mean(data2)],
                    color='k', linewidth=2, zorder=1)
    if showmedians:
        vp['cmedians'].set_color('k')
        vp['cmedians'].set_linewidth(2)
        ax.scatter(1.1, np.median(data1), 
                   s=30, c=colour1, ec='none', lw=.5, zorder=2)
        ax.scatter(1.9, np.median(data2), 
                   s=30, c=colour2, ec='none', lw=.5, zorder=2)
        if paired:
            ax.plot([1.1, 1.9], 
                    [data1, data2], 
                    color='grey', alpha=.05, linewidth=1, zorder=1)
            ax.plot([1.1, 1.9], [np.median(data1), np.median(data2)],
                    color='k', linewidth=2, zorder=1)
    for i in [0,1]:
        vp['bodies'][i].set_edgecolor('none')
        vp['bodies'][i].set_alpha(.75)
        b = vp['bodies'][i]
        # get the centre 
        m = np.mean(b.get_paths()[0].vertices[:,0])
        # make paths not go further right/left than the centre 
        if i==0:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
        if i==1:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

    ax.scatter([1.1]*len(data1), 
               data1, 
               s=10, c=colour1, ec='none', lw=.5, alpha=.05)
    ax.scatter([1.9]*len(data2), 
               data2, 
               s=10, c=colour2, ec='none', lw=.5, alpha=.05)
    
    y_range = [max(max(data1), max(data2)), min(min(data1), min(data2))]
    y_range_tot = y_range[0]-y_range[1]
        
    if statistics:
        if paired:
            wilc_stat, wilc_p = wilcoxon(data1, data2)
            ttest_stat, ttest_p = ttest_rel(data1, data2)
            ax.plot([1, 2], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
            ax.text(1.5, y_range[0]+y_range_tot*.05, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
                    ha='center', va='bottom', color='k', fontsize=8)
        else:
            rank_stat, rank_p = ranksums(data1, data2)
            ttest_stat, ttest_p = ttest_ind(data1, data2)
            ax.plot([1, 2], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
            ax.text(1.5, y_range[0]+y_range_tot*.05, 'ranksums_p={}\nttest_p={}'.format(round(rank_p, 5), round(ttest_p, 5)), 
                    ha='center', va='bottom', color='k', fontsize=8)
        
    ax.set(xticks=[1,2], xticklabels=xticklabels, xlim=(.5, 2.5),
           ylabel=ylabel,
           title=title)
    
    if yscale!=None:
        ax.set_yscale('symlog')
    
    for s in ['top', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
        
    fig.tight_layout()
    plt.grid(False)
    plt.show()
    
    if save:
        fig.savefig(savepath+'.png',
                    dpi=dpi,
                    bbox_inches='tight')
        fig.savefig(savepath+'.pdf',
                    bbox_inches='tight')
        
        
#%% profile-plotting functions
def scale_min_max(mean_data, sem_data=[]):
    if len(sem_data)==0:
        full_range = max(mean_data)-min(mean_data)
        max_point = max(mean_data); min_point = min(mean_data)
    else:
        full_range = max(mean_data+sem_data)-min(mean_data-sem_data)
        max_point = max(mean_data+sem_data); min_point = min(mean_data-sem_data)
    scale_max = max_point+full_range*.05
    scale_min = min_point-full_range*.05
    
    return (scale_min, scale_max)