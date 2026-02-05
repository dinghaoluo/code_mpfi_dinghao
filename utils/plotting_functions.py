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
from scipy.stats import wilcoxon, ranksums, ttest_rel, ttest_ind, sem, ttest_1samp


#%% functions 
def add_scale_bar(ax, x_start, y_start, x_len, y_len, color='k', lw=2):
    # horizontal scale bar (time)
    ax.plot([x_start, x_start + x_len], [y_start, y_start], color=color, lw=lw, solid_capstyle='butt')
    # vertical scale bar (dF/F)
    ax.plot([x_start, x_start], [y_start, y_start + y_len], color=color, lw=lw, solid_capstyle='butt')


# -------------------------------
# BAR PLOT WITH SCATTER (PAIRED)
# -------------------------------
def plot_bar_with_paired_scatter(
        ax, ctrl_vals, stim_vals, colors=('grey', 'firebrick'),
        title='', ylabel='% cells', xticklabels=('ctrl.', 'stim.'),
        ylim=None
        ):
    """
    plot paired bar means with sem, individual paired scatter points,
    and paired statistical comparisons.
    
    parameters:
    - ax: matplotlib axis
        axis to draw the plot on
    - ctrl_vals: array-like
        control condition values (paired with stim_vals)
    - stim_vals: array-like
        stimulation condition values
    - colors: tuple
        colours for control and stimulation bars
    - title: str
        plot title
    - ylabel: str
        y-axis label
    - xticklabels: tuple
        x-axis category labels
    - ylim: tuple or none
        y-axis limits
    """
    
    def sigstars(p):
        return 'ns' if p >= 0.05 else ('*' if p < 0.05 and p >= 0.01 else ('**' if p < 0.01 and p >= 0.001 else ('***' if p < 0.001 and p >= 1e-4 else '****')))
    def annotate(ax, x1, x2, y, text, yrange):
        bump = 0.01 * yrange
        ax.plot([x1, x1, x2, x2], [y-0.5*bump, y, y, y-0.5*bump], lw=0.8, color='k')
        ax.text((x1+x2)/2, y + 0.5*bump, text, ha='center', va='bottom', fontsize=6)
    
    ctrl_vals = np.asarray(ctrl_vals, dtype=float)
    stim_vals = np.asarray(stim_vals, dtype=float)
    mask = np.isfinite(ctrl_vals) & np.isfinite(stim_vals)
    ctrl = ctrl_vals[mask]
    stim = stim_vals[mask]
    assert len(ctrl) == len(stim) and len(ctrl) > 0, 'need ≥1 paired finite values'

    # bars: mean ± sem
    means = [np.nanmean(ctrl), np.nanmean(stim)]
    errs  = [sem(ctrl, nan_policy='omit'), sem(stim, nan_policy='omit')]

    barc = ax.bar(
        [0, 1], means, yerr=errs, capsize=2, width=0.6, edgecolor='none',
        alpha=.6, zorder=2,
        error_kw={'elinewidth': 0.6, 'capthick': 0.6, 'ecolor': 'k'}
    )
    barc.patches[0].set_facecolor(colors[0])
    barc.patches[1].set_facecolor(colors[1])

    # paired points + connecting lines (no jitter)
    x0 = np.zeros(len(ctrl))
    x1 = np.ones(len(stim))
    for y0, y1 in zip(ctrl, stim):
        ax.plot([0, 1], [y0, y1], lw=0.6, color='k', alpha=.3, zorder=3)
    ax.scatter(x0, ctrl, s=8, color=colors[0], edgecolor='none', alpha=.5, zorder=4)
    ax.scatter(x1, stim, s=8, color=colors[1], edgecolor='none', alpha=.5, zorder=4)

    # axes + limits
    ylims = (0, 100) if ylim is None else ylim
    ax.set(xticks=[0,1], xticklabels=xticklabels, ylabel=ylabel, title=title, ylim=ylims)
    for s in ['top','right']:
        ax.spines[s].set_visible(False)

    # stats (paired)
    try:
        w_stat, w_p = wilcoxon(ctrl, stim, alternative='two-sided', zero_method='wilcox', mode='auto')
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    t_stat, t_p = ttest_rel(ctrl, stim)
    
    rank_sum_stat, rank_sum_p = ranksums(ctrl, stim)
    
    t_ind_stat, t_ind_p = ttest_ind(ctrl, stim)

    # annotations
    yrange = ylims[1] - ylims[0]
    top_data_val = max(np.nanmax(ctrl), np.nanmax(stim))
    top_bar_val = max(means[0] + (errs[0] if np.isfinite(errs[0]) else 0),
                      means[1] + (errs[1] if np.isfinite(errs[1]) else 0))
    top_y = max(top_data_val, top_bar_val)

    y1 = top_y
    y2 = y1
    annotate(ax, 0, 1, y1, f"Wilcoxon p={w_p:.4g} ({sigstars(w_p)})", yrange)
    annotate(ax, 0, 1, y2, f"t-test p={t_p:.4g} ({sigstars(t_p)})", yrange)
    ax.set_ylim(ylims[0], max(ylims[1], y2 + 0.08 * yrange))

    print(f'Wilc p = {w_p}\nttest p = {t_p}')
# ------------------------------------
# BAR PLOT WITH SCATTER (PAIRED) ENDS
# ------------------------------------


# -------------------------------
# BOX PLOT WITH SCATTER (PAIRED)
# -------------------------------
def plot_box_with_scatter(ctrl_data, stim_data, xlabel, savepath, 
                          title='', show_scatter=True,
                          ctrl_color='grey', stim_color='royalblue'):
    """
    plot horizontal paired boxplots with optional scatter overlay.
    
    parameters:
    - ctrl_data: array-like
        control condition values
    - stim_data: array-like
        stimulation condition values
    - xlabel: str
        x-axis label
    - savepath: str
        file path (without extension) for saving
    - title: str
        plot title
    - show_scatter: bool
        whether to overlay individual data points
    - ctrl_color: str or tuple
        colour for control box
    - stim_color: str or tuple
        colour for stimulation box
    """
    fig, ax = plt.subplots(figsize=(2.6, 1.4))

    boxplots = ax.boxplot(
        [stim_data, ctrl_data],  # stim first to match colour order
        vert=False,
        positions=[2, 1],
        widths=0.25,
        patch_artist=True,
        medianprops={'color': 'black'},
        capprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        flierprops={'marker': 'o', 'color': 'black', 'markersize': 3}
    )

    for patch, color in zip(boxplots['boxes'], (stim_color, ctrl_color)):
        patch.set_facecolor(color)
        patch.set_alpha(1)
    
    if show_scatter:
        ax.scatter(stim_data, np.full_like(stim_data, 1.7),
                   s=20, c=stim_color, alpha=0.75, ec='none')
        ax.scatter(ctrl_data, np.full_like(ctrl_data, 1.3),
                   s=20, c=ctrl_color, alpha=0.75, ec='none')

    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

    ax.set(
        title=title,
        xlabel=xlabel,
        yticks=(1, 2),
        yticklabels=('ctrl.', 'stim.'),
        ylim=(0.6, 2.4)
    )

    for ext in ('.png', '.pdf'):
        fig.savefig(f'{savepath}{ext}', dpi=300, bbox_inches='tight')

    plt.close(fig)
# ------------------------------------
# BOX PLOT WITH SCATTER (PAIRED) ENDS
# ------------------------------------


# ------------------------
# VIOLINPLOT WITH SCATTER
# ------------------------
def plot_violin_with_scatter(data0, data1, colour0, colour1,
                             paired=True, alpha=.25,
                             xlabel=' ', xticks=[1,2], xticklabels=['data0', 'data1'],
                             ylabel=' ', yscale=None,
                             xlim=None, ylim=None,
                             title=' ',
                             showscatter=False, showmainline=True, showline=True,
                             showmeans=False, showmedians=True, showextrema=False,
                             print_statistics=False, plot_statistics=True,
                             save=False, savepath=' ', dpi=300,
                             figsize=(1.8,2.2),
                             pngonly=False):
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
    
    fig, ax = plt.subplots(figsize=figsize)

    vp = ax.violinplot([data0, data1],
                       positions=[1.1,1.9],
                       showmeans=showmeans, showmedians=showmedians, 
                       showextrema=False)

    vp['bodies'][0].set_color(colour0)
    vp['bodies'][1].set_color(colour1)
    if showmeans:
        vp['cmeans'].set_color('k')
        vp['cmeans'].set_linewidth(2)
        ax.scatter(1.25, np.mean(data0), 
                   s=30, c=colour0, ec='none', lw=.5, zorder=3)
        ax.scatter(1.75, np.mean(data1), 
                   s=30, c=colour1, ec='none', lw=.5, zorder=3)
        if paired and showline:
            ax.plot([1.25, 1.75], 
                    [data0, data1], 
                    color='grey', alpha=alpha, linewidth=1, zorder=1)
        if showmainline:
            ax.plot([1.25, 1.75], [np.mean(data0), np.mean(data1)],
                    color='k', linewidth=2, zorder=2)
    if showmedians:
        vp['cmedians'].set_color('k')
        vp['cmedians'].set_linewidth(2)
        ax.scatter(1.25, np.median(data0), 
                   s=30, c=colour0, ec='none', lw=.5, zorder=2)
        ax.scatter(1.75, np.median(data1), 
                   s=30, c=colour1, ec='none', lw=.5, zorder=2)
        if paired and showline:
            ax.plot([1.25, 1.75], 
                    [data0, data1], 
                    color='grey', alpha=alpha, linewidth=1, zorder=1)
        if showmainline:
            ax.plot([1.25, 1.75], [np.median(data0), np.median(data1)],
                    color='k', linewidth=2, zorder=1)
    for i in [0,1]:
        vp['bodies'][i].set_edgecolor('none')
        vp['bodies'][i].set_alpha(.75)
        b = vp['bodies'][i]
        m = np.mean(b.get_paths()[0].vertices[:,0])
        if i==0:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], -np.inf, m)
        if i==1:
            b.get_paths()[0].vertices[:,0] = np.clip(b.get_paths()[0].vertices[:,0], m, np.inf)

    if showscatter:
        ax.scatter([1.25]*len(data0), data0, s=10, c=colour0, ec='none', lw=.5, alpha=alpha)
        ax.scatter([1.75]*len(data1), data1, s=10, c=colour1, ec='none', lw=.5, alpha=alpha)

    # --- NEW SECTION: print mean ± sem above violins ---
    mean0, mean1 = np.nanmean(data0), np.nanmean(data1)
    sem0, sem1   = sem(data0, nan_policy='omit'), sem(data1, nan_policy='omit')
    
    med0, med1 = np.nanmedian(data0), np.nanmedian(data1)
    q25_0, q75_0 = np.percentile(data0, [25, 75])
    q25_1, q75_1 = np.percentile(data1, [25, 75])
    
    y_max = max(np.nanmax(data0), np.nanmax(data1))
    y_min = min(np.nanmin(data0), np.nanmin(data1))
    y_offset = (ylim[1] - ylim[0]) * 0.05 if ylim else 0.05 * (y_max - y_min)
    
    ax.text(
        1.1, y_max + y_offset,
        f'Med = {med0:.2f}\n'
        f'IQR = [{q25_0:.2f}, {q75_0:.2f}]\n'
        f'{mean0:.2f} ± {sem0:.2f}',
        ha='center', va='bottom', fontsize=7, color=colour0
    )
    
    ax.text(
        1.9, y_max + y_offset,
        f'Med = {med1:.2f}\n'
        f'IQR = [{q25_1:.2f}, {q75_1:.2f}]\n'
        f'{mean1:.2f} ± {sem1:.2f}',
        ha='center', va='bottom', fontsize=7, color=colour1
    )

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

    # [statistical tests below unchanged]
    if paired:
        wilc_stat, wilc_p = wilcoxon(data0, data1)
        ttest_stat, ttest_p = ttest_rel(data0, data1)
        wilc_p_str = '{:.2e}'.format(wilc_p)
        ttest_p_str = '{:.2e}'.format(ttest_p)
        if print_statistics:
            print('\ndata 0:')
            print(f'  median = {med0:.4g}')
            print(f'  IQR    = [{q25_0:.4g}, {q75_0:.4g}]')
            print(f'  mean ± SEM = {mean0:.4g} ± {sem0:.4g}')
        
            print('data 1:')
            print(f'  median = {med1:.4g}')
            print(f'  IQR    = [{q25_1:.4g}, {q75_1:.4g}]')
            print(f'  mean ± SEM = {mean1:.4g} ± {sem1:.4g}')
        
            print(f'wilcoxon: W={wilc_stat:.4g}, p={wilc_p_str}')
            print(f'ttest:    t={ttest_stat:.4g}, p={ttest_p_str}')
        if plot_statistics:
            ax.plot([1.1, 1.9], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
            ax.text(1.5, y_range[0]+y_range_tot*.05, 
                    f'wilc_p={wilc_p_str}\nttest_p={ttest_p_str}', 
                    ha='center', va='bottom', color='k', fontsize=8)
    else:
        wilc_stat, wilc_p = ranksums(data0, data1)
        ttest_stat, ttest_p = ttest_ind(data0, data1)
        wilc_p_str = '{:.2e}'.format(wilc_p)
        ttest_p_str = '{:.2e}'.format(ttest_p)
        if print_statistics:
            print('\ndata 0:')
            print(f'  median = {med0:.4g}')
            print(f'  IQR    = [{q25_0:.4g}, {q75_0:.4g}]')
            print(f'  mean ± SEM = {mean0:.4g} ± {sem0:.4g}')
        
            print('data 1:')
            print(f'  median = {med1:.4g}')
            print(f'  IQR    = [{q25_1:.4g}, {q75_1:.4g}]')
            print(f'  mean ± SEM = {mean1:.4g} ± {sem1:.4g}')
        
            print(f'ranksums: z={wilc_stat:.4g}, p={wilc_p_str}')
            print(f'ttest:    t={ttest_stat:.4g}, p={ttest_p_str}')
        if plot_statistics:
            ax.plot([1.1, 1.9], [y_range[0]+y_range_tot*.05, y_range[0]+y_range_tot*.05], c='k', lw=.5)
            ax.text(1.5, y_range[0]+y_range_tot*.05, 
                    f'ranksums_p={wilc_p_str}\nttest_p={ttest_p_str}', 
                    ha='center', va='bottom', color='k', fontsize=8)

    ax.set(xticks=[1.1,1.9], xticklabels=xticklabels,
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
        if pngonly:
            fig.savefig(f'{savepath}.png',
                        dpi=dpi,
                        bbox_inches='tight')
        else:
            for ext in ['.png', '.pdf']:
                fig.savefig(f'{savepath}{ext}',
                            dpi=dpi,
                            bbox_inches='tight')
# -----------------------------
# VIOLINPLOT WITH SCATTER ENDS
# -----------------------------


# ------
# ECDFs
# ------
def plot_ecdfs(data0, data1, 
               title=' ',
               xlabel=' ', ylabel='cumulative probability',
               legend_labels=[' ', ' '],
               colours=['lightcoral', 'firebrick'],
               save=False, savepath=' ', dpi=300, figsize=(2.4, 3.0),
               stats_text=None):
    """
    plot ECDFs for two datasets with optional stats annotation.

    parameters:
    - data0, data1: arrays of data values
    - title, xlabel, ylabel: str, labels
    - legend_labels: list of str, legend entries for data0 and data1
    - colours: list of str or colour tuples, colours for the two curves
    - save: bool, if true, saves plot to savepath
    - stats_text: str, optional stats annotation printed inside the plot
    - dpi, figsize: plot formatting

    returns:
    - none
    """

    data0 = np.sort(data0)
    data1 = np.sort(data1)

    x0 = np.concatenate([[data0[0]], data0])
    x1 = np.concatenate([[data1[0]], data1])
    y0 = np.concatenate([[0], np.arange(1, len(data0)+1) / len(data0)])
    y1 = np.concatenate([[0], np.arange(1, len(data1)+1) / len(data1)])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x0, y0, label=legend_labels[0], color=colours[0])
    ax.plot(x1, y1, label=legend_labels[1], color=colours[1])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6)

    # Add stats annotation if provided
    if stats_text is not None:
        ax.text(0.97, 0.03, stats_text,
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    plt.show()

    if save:
        for ext in ['.png', '.pdf']:
            fig.savefig(f'{savepath}{ext}', dpi=dpi, bbox_inches='tight')
# -----------
# ECDFs ENDS
# -----------


# ------------------
# SINGLE VIOLINPLOT
# ------------------
def plot_single_violin(data, colour, ref_val=0, 
                       xpos=1, xticklabels=['stim.'],
                       ylabel=' ', ylim=None,
                       title=' ',
                       showscatter=True,
                       alpha=0.5,
                       figsize=(1.7, 2.2),
                       plot_statistics=True,
                       print_statistics=False,
                       save=False, savepath=' ',
                       dpi=300,
                       pngonly=False):
    """
    plot a single violin with scatter and one-sample statistical tests.

    parameters:
    - data: array-like
        values to plot
    - colour: str or tuple
        violin and scatter colour
    - ref_val: float
        reference value for one-sample tests (default: 0)
    - xpos: float
        x position of the violin (default: 1)
    - xticklabels: list
        x-axis labels
    - ylabel: str
        y-axis label
    - ylim: tuple or None
        y-axis limits
    - title: str
        plot title
    - showscatter: bool
        show individual data points
    - alpha: float
        scatter/violin transparency
    - figsize: tuple
        figure size
    - plot_statistics: bool
        write statistics on the plot
    - print_statistics: bool
        print statistics to console
    - save: bool
        save figure
    - savepath: str
        save path without extension
    - dpi: int
        resolution
    - pngonly: bool
        save only png if true
    """

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # statistics
    tval, p_t = ttest_1samp(data, ref_val)
    wstat, p_w = wilcoxon(data - ref_val)

    mean_v = np.mean(data)
    sem_v  = sem(data)
    med_v  = np.median(data)
    q25, q75 = np.percentile(data, [25, 75])

    ymax, ymin = np.max(data), np.min(data)
    y_range = ymax - ymin

    if print_statistics:
        print(f'median = {med_v:.4g}')
        print(f'IQR = [{q25:.4g}, {q75:.4g}]')
        print(f'mean ± SEM = {mean_v:.4g} ± {sem_v:.4g}')
        print(f'ttest: t={tval:.4g}, p={p_t:.2e}')
        print(f'wilcoxon: W={wstat:.4g}, p={p_w:.2e}')

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(
        data, positions=[xpos],
        showmeans=False, showmedians=True, showextrema=False
    )

    for pc in parts['bodies']:
        pc.set_facecolor(colour)
        pc.set_edgecolor('none')
        pc.set_alpha(alpha)

    parts['cmedians'].set_color('k')
    parts['cmedians'].set_linewidth(1.2)

    if showscatter:
        ax.scatter(
            np.full(len(data), xpos), data,
            s=10, c=colour, ec='none',
            alpha=alpha, zorder=3
        )

    # summary text (top)
    ax.text(
        xpos,
        ymax + 0.05 * y_range,
        f'Med = {med_v:.2f}\n'
        f'IQR = [{q25:.2f}, {q75:.2f}]\n'
        f'{mean_v:.2f} ± {sem_v:.2f}',
        ha='center', va='bottom',
        fontsize=7, color=colour
    )

    # stats text (bottom)
    if plot_statistics:
        ax.text(
            xpos,
            ymin - 0.12 * y_range,
            f't(1-samp)={tval:.2f}, p={p_t:.2e}\n'
            f'Wilcoxon={wstat:.2f}, p={p_w:.2e}',
            ha='center', va='top',
            fontsize=6.5, color='k'
        )

    ax.set(
        xlim=(xpos - 0.5, xpos + 0.5),
        xticks=[xpos],
        xticklabels=xticklabels,
        ylabel=ylabel,
        title=title
    )

    if ylim is not None:
        ax.set(ylim=ylim)

    for s in ['top', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    plt.grid(False)
    plt.show()

    if save:
        if pngonly:
            fig.savefig(f'{savepath}.png',
                        dpi=dpi, bbox_inches='tight')
        else:
            for ext in ['.png', '.pdf']:
                fig.savefig(f'{savepath}{ext}',
                            dpi=dpi, bbox_inches='tight')
# -----------------------
# SINGLE VIOLINPLOT ENDS
# -----------------------

        
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