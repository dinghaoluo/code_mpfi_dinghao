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
def plot_violin_with_scatter(data0, data1, colour0, colour1,
                             paired=True, alpha=.25,
                             xlabel=' ', xticks=[1,2], xticklabels=['data0', 'data1'],
                             ylabel=' ', yscale=None,
                             xlim=None, ylim=None,
                             title=' ',
                             showscatter=True, showmainline=True,
                             showmeans=False, showmedians=True, showextrema=False,
                             print_statistics=True, plot_statistics=True,
                             save=False, savepath=' ', dpi=120,
                             figsize=(1.8,2.4)):
    """
    plot half-violins with optional scatter and statistical comparisons

    parameters
    ----------
    data0 : array-like
        values for the first dataset (plotted at x=1)
    data1 : array-like
        values for the second dataset (plotted at x=2)
    colour0 : str or tuple
        colour for the first dataset
    colour1 : str or tuple
        colour for the second dataset
    paired : bool, optional
        if true, use paired statistical tests; otherwise, use unpaired tests (default: True)
    alpha : float, optional
        transparency level for scatter points and lines (default: 0.25)
    xlabel : str, optional
        label for the x-axis (default: ' ')
    xticks : list, optional
        positions for x-axis ticks (default: [1, 2])
    xticklabels : list, optional
        labels for the x-axis ticks (default: ['data0', 'data1'])
    ylabel : str, optional
        label for the y-axis (default: ' ')
    yscale : str, optional
        scale for the y-axis (e.g., 'symlog'); if none, uses linear scale (default: None)
    xlim : tuple, optional
        limits for the x-axis (default: None)
    ylim : tuple, optional
        limits for the y-axis (default: None)
    title : str, optional
        title for the plot (default: ' ')
    showscatter : bool, optional
        if true, scatter individual data points on the plot (default: True)
    showmainline : bool, optional
        if true, draw a line connecting mean or median values (default: True)
    showmeans : bool, optional
        if true, display mean markers and lines (default: False)
    showmedians : bool, optional
        if true, display median markers and lines (default: True)
    showextrema : bool, optional
        if true, show extrema for violins (default: False)
    print_statistics : bool, optional
        if true, print statistical results in the console (default: True)
    plot_statistics : bool, optional
        if true, display statistical results on the plot (default: True)
    save : bool, optional
        if true, save the plot as a .png and .pdf file (default: False)
    savepath : str, optional
        path to save the plot (default: ' ')
    dpi : int, optional
        resolution for the saved image (default: 120)
    figsize  : tuple, optional 
        figure size (default: (1.8, 2.4))

    returns
    -------
    none

    notes
    -----
    - half-violins are plotted for two datasets at x=1 and x=2
    - supports paired or unpaired statistical tests (wilcoxon and paired t-tests for paired data; rank-sum and unpaired t-tests for unpaired data)
    - scatter points show individual data values; mean or median markers and lines are optional
    - violins are coloured based on `colour0` and `colour1`
    - saves the plot in both .png and .pdf formats if `save=True`
    """
    
    print(data1)
    fig, ax = plt.subplots(figsize=figsize)
    
    vp = ax.violinplot([data0, data1],
                       positions=[1,2],
                       showmeans=showmeans, showmedians=showmedians, 
                       showextrema=False)

    vp['bodies'][0].set_color(colour0)
    vp['bodies'][1].set_color(colour1)
    if showmeans:
        vp['cmeans'].set_color('k')
        vp['cmeans'].set_linewidth(2)
        ax.scatter(1.1, np.mean(data0), 
                   s=30, c=colour0, ec='none', lw=.5, zorder=3)
        ax.scatter(1.9, np.mean(data1), 
                   s=30, c=colour1, ec='none', lw=.5, zorder=3)
        if paired:
            ax.plot([1.1, 1.9], 
                    [data0, data1], 
                    color='grey', alpha=alpha, linewidth=1, zorder=1)
        if showmainline:
            ax.plot([1.1, 1.9], [np.mean(data0), np.mean(data1)],
                    color='k', linewidth=2, zorder=2)
    if showmedians:
        vp['cmedians'].set_color('k')
        vp['cmedians'].set_linewidth(2)
        ax.scatter(1.1, np.median(data0), 
                   s=30, c=colour0, ec='none', lw=.5, zorder=2)
        ax.scatter(1.9, np.median(data1), 
                   s=30, c=colour1, ec='none', lw=.5, zorder=2)
        if paired:
            ax.plot([1.1, 1.9], 
                    [data0, data1], 
                    color='grey', alpha=alpha, linewidth=1, zorder=1)
        if showmainline:
            ax.plot([1.1, 1.9], [np.median(data0), np.median(data1)],
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

    if showscatter:
        ax.scatter([1.1]*len(data0), 
                   data0, 
                   s=10, c=colour0, ec='none', lw=.5, alpha=alpha)
        ax.scatter([1.9]*len(data1), 
                   data1, 
                   s=10, c=colour1, ec='none', lw=.5, alpha=alpha)
    
    if xlim is not None:
        ax.set(xlim=xlim)
    else:
        ax.set(xlim=(.5, 2.5))
    if ylim is not None:
        ax.set(ylim=ylim)
        y_range = (ylim[1], ylim[0])
        y_range_tot = ylim[1]-ylim[0]
    else:
        y_range = (max(max(data0), max(data1)), min(min(data0), min(data1)))
        y_range_tot = y_range[0]-y_range[1]

    if paired:
        wilc_stat, wilc_p = wilcoxon(data0, data1)
        ttest_stat, ttest_p = ttest_rel(data0, data1)
        if print_statistics:
            print(f'wilc: {wilc_stat}, p={wilc_p}')
        if plot_statistics:
            ax.plot([1, 2], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
            ax.text(1.5, y_range[0]+y_range_tot*.05, 'wilc_p={}\nttest_p={}'.format(round(wilc_p, 5), round(ttest_p, 5)), 
                    ha='center', va='bottom', color='k', fontsize=8)
    else:
        rank_stat, rank_p = ranksums(data0, data1)
        ttest_stat, ttest_p = ttest_ind(data0, data1)
        if print_statistics:
            print(f'ranksums: {rank_stat}, p={rank_p}')
        if plot_statistics:
            ax.plot([1, 2], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
            ax.text(1.5, y_range[0]+y_range_tot*.05, 'ranksums_p={}\nttest_p={}'.format(round(rank_p, 5), round(ttest_p, 5)), 
                    ha='center', va='bottom', color='k', fontsize=8)
        
    ax.set(xticks=[1,2], xticklabels=xticklabels,
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
        for ext in ['.png', '.pdf']:
            fig.savefig(f'{savepath}{ext}',
                        dpi=dpi,
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

def get_lower_upper_bounds_violin(body):
    '''take the body of a violinplot'''
    '''return (lower_bound, upper_bound)'''
    vertices = body.get_paths()[0].vertices 
    y_values = vertices[:,1]
    
    return min(y_values), max(y_values)