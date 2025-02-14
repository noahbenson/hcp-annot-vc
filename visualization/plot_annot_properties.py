import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import utils

rc = {'text.color': 'black',
      'axes.labelcolor': 'black',
      'xtick.color': 'black',
      'ytick.color': 'black',
      'xtick.labelcolor': 'black',
      'ytick.labelcolor': 'black',
      'font.family': 'helveticaneue',
      'font.weight': 'light',
      'font.size' : 11,
      'figure.dpi': 72*3,
      'savefig.dpi': 72*4,
      }
mpl.rcParams.update(rc)

def calculate_percent(roi, cortex):
    return roi*100/cortex

def melt_rois_to_lh_rh(df, roi, id_vars=['sid','anatomist']):
    original_roi_cols = [col for col in df if roi in col]
    df = df[original_roi_cols + id_vars]
    hemi_cols = [col[3].lower()+'h' for col in original_roi_cols]
    df = df.rename(columns=dict(zip(original_roi_cols, hemi_cols)))
    df = pd.melt(df, id_vars=id_vars, value_vars=hemi_cols, var_name='hemisphere', value_name=roi, ignore_index=True)
    return df

def melt_roi_list_to_lh_rh(df, roi_list, id_vars=['sid','anatomist']):
    tmp = {}
    for roi in roi_list:
        tmp[roi] = melt_rois_to_lh_rh(df, roi, id_vars=id_vars)
    dfs = [tmp[roi].set_index(id_vars + ['hemisphere']) for roi in roi_list]
    long_df = pd.concat(dfs, axis=1).reset_index()
    return long_df

def unmelt_lh_rh_rois(df, roi_list, id_vars=['sid','anatomist']):
    wide_df = {}
    for hemi in ['lh', 'rh']:
        tmp = df.query('hemisphere == @hemi')
        col_names = [col for col in tmp if col in roi_list]
        hemi_col_names = [f'{hemi}{roi}' for roi in col_names]
        tmp = tmp.rename(columns=dict(zip(col_names, hemi_col_names)))
        wide_df[hemi] = tmp.drop(columns='hemisphere')

    wide_df = pd.merge(wide_df['lh'], wide_df['rh'], on=id_vars)
    return wide_df


def get_correlation_matrix(df, k=0):
    corr_matrix = df.corr()
    mask = np.triu(corr_matrix, k=k)
    return corr_matrix, mask


def heatmap_surface_area(df, mask=None, ax=None, cmap="YlOrRd", font_scale=1,
                         annot=True, boundary_line=None, width=5, height=1, cbar=True,
                         fmt=".1f", vmin=0, vmax=1, save_path=None, rc=None, **kwarg):
    sns.set_theme(context="notebook", style='ticks', rc=rc, font_scale=font_scale)
    if annot is True:
        annot_kws = {"size": rc['font.size']* font_scale * 0.6}
    else:
        annot_kws = None
    ax = sns.heatmap(df, mask=mask, 
                     annot=annot, annot_kws=annot_kws, ax=ax, fmt=fmt, cbar=cbar,
                     cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .7},
                     linewidth=.3, square=True)
    # Get current ticks and labels
    yticks = ax.get_yticks()
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]

    # Remove the first and the last xy tick and label
    ax.set_yticks(yticks[1:])
    ax.set_yticklabels(yticklabels[1:])
    
    xticks = ax.get_xticks()
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]

    ax.set_xticks(xticks[:-1])
    ax.set_xticklabels(xticklabels[:-1])

    if boundary_line is not None:
        ax.hlines(boundary_line, ax.get_xlim()[0], (ax.get_xlim()[1]/2), color='blue', linewidth=rc['xtick.major.width']*1.2, linestyles='--'),
        ax.vlines(boundary_line, (ax.get_ylim()[0]/2), ax.get_ylim()[0], color='blue', linewidth=rc['xtick.major.width']*1.2, linestyles='--')
    if save_path is not None:
        parent_path = Path(save_path)
        if not os.path.exists(parent_path.parent.absolute()):
            os.makedirs(parent_path.parent.absolute())
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    return ax


def ax_violinplot_surface_area(ax, df, x, y, order, 
                               cmap=None, rc=None, ylabel=None,
                               hue='hemisphere', hue_order=['lh','rh'], 
                               split=True, bw=.2, linewidth=.5, **kwargs):
    sns.despine(top=True, bottom=True, right=True, left=False)
    sns.set_theme(context='notebook', style='ticks', rc=rc)
    ax = sns.violinplot(df, x=x, y=y, split=True,
                           order=order, density_norm="width",
                           hue=hue, hue_order=hue_order, bw=bw,
                           palette=cmap, linewidth=linewidth, ax=ax, **kwargs)
    if ylabel is not None:
        ax.set(ylabel = ylabel)
    return ax

import matplotlib.pyplot as plt
import seaborn as sns

def plot_violin_surface_area(plot_df, x='gender', order=['M', 'F'], 
                             y='percent', hue='hemisphere', 
                             hue_order=['LH', 'RH'], rc=None, 
                             rois=['V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2'], 
                             ylabel='Relative surface area (%)', save_path=None):
    """
    Plots violin plots for relative surface area of different ROIs,
    with specific styling for hemisphere and gender differences.

    Parameters:
    - plot_df: DataFrame containing the data to plot. 
    - x: column of plot_df that will be plotted on x axis,
    - y: a column of plot_df that is plotted on y axis,
    - violin_rc: Dictionary of parameters for the violin plot.
    """
    # Define color palettes
    hemi_palette = sns.color_palette(["#6a0dad", "#2ca02c"])

    # Create subplots
    fig, axes = plt.subplots(1, 6, figsize=(7, 2.5), sharey=False)
    fig.text(0.5, 0, "ROIs", ha="center")
    fig.text(0.165, 0.83 , "LH", fontsize=rc['font.size']*0.7, 
             fontweight="bold", fontname='Arial', color=hemi_palette[0], ha="center")
    fig.text(0.195, 0.83 , "RH", fontname='Arial', fontsize=rc['font.size']*0.7,
             fontweight="bold", color=hemi_palette[1], ha="center")

    # Loop through axes and ROIs to create plots
    for ax, roi in zip(axes, rois):
        tmp = plot_df.query('ROIs == @roi')

        ax_violinplot_surface_area(ax=ax, df=tmp, rc=rc, ylabel=ylabel,
                                   hue=hue, hue_order=hue_order,
                                   x=x, order=order, 
                                   y=y, inner='stick',
                                   bw=.2, fill=False,
                                   cmap=hemi_palette, linewidth=0.5)

        ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        ax.set_title(roi)
        ax.set(ylim=[0, 2.5], yticks=[0, 0.5, 1, 1.5, 2, 2.5])
        
        grouped = tmp.groupby([x,hue])[y]
        grouped_medians = grouped.median().unstack()  # Compute median for each gender
        # Compute and plot the interquartile range (IQR)
        q1 = grouped.quantile(0.25).unstack()  # 25th percentile (Q1)
        q3 = grouped.quantile(0.75).unstack()  # 75th percentile (Q3)    
        # X-axis positions ('M' at 0, 'F' at 1), LH shifted slightly left, RH slightly right
        x_offsets = {hue_order[0]: -0.1, hue_order[1]: 0.1}  # Small offset to separate LH and RH dots
        for gender, x_pos in zip(order, [0, 1]):
            for hemisphere in hue_order:
                median_value = grouped_medians.loc[gender, hemisphere]
                ax.scatter(x_pos + x_offsets[hemisphere], median_value, 
                           color='k', s=4, zorder=3)
                lower = q1.loc[gender, hemisphere]
                upper = q3.loc[gender, hemisphere]
                ax.vlines(x_pos + x_offsets[hemisphere], 
                          ymin=lower, ymax=upper, color='k', linewidth=0.5)
        
    # Customize second to sixth subplot y-axis
    for ax in axes[1:]:
        ax.yaxis.set_ticks([])  # Remove y-axis ticks
        ax.yaxis.label.set_visible(False)
        ax.spines["left"].set_linestyle((0, (6, 10)))  # Custom dotted y-axis
        ax.spines["left"].set_color("grey")  # Keep visible if needed
        

    # Adjust figure layout
    plt.subplots_adjust(bottom=0.15)
    

    if save_path is not None:
        parent_path = Path(save_path)
        if not os.path.exists(parent_path.parent.absolute()):
            os.makedirs(parent_path.parent.absolute())
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    return grouped



# def violinplot_surface_area(df, x, y, x_order, hue='hemisphere', hue_order=['lh','rh'], split=True,
#                             col=None, col_wrap=None, bw=.2, linewidth=0.5, font_size=11,
#                             width=3.14, height=3, cmap=sns.color_palette("Spectral"), save_path=None):
#     rc.update({'axes.labelpad': 10, 'figure.figsize':(width, height),'font.size' : font_size})
#     utils.set_rcParams(rc)
#     sns.set_theme(context="notebook", style='ticks', rc=rc)
#     sns.despine(top=True, bottom=True, right=True)
#     if 'percent' in y:
#         y_label = 'Relative surface area (%)'
#     elif 'mm2' in y:
#         y_label = r'Surface area ($mm^2$)'
#     grid = sns.FacetGrid(df,
#                          col=col, col_wrap=col_wrap,
#                          legend_out=True, 
#                          sharex=True, sharey=True)
#     grid = grid.map(sns.violinplot, x, y, hue,
#                     hue_order=hue_order, split=split, order=x_order, palette=cmap, cut=0,
#                     inner='box', linewidth=linewidth, saturation=0.9, bw=bw, edgecolor='black')
#     grid.add_legend(bbox_to_anchor=(1, 0.8))
#     grid.set_axis_labels('ROIs', y_label)
#     if col is not None:
#         for subplot_title, ax in grid.axes_dict.items():
#             ax.set_title(f"{subplot_title.title()}")
#     if save_path is not None:
#         parent_path = Path(save_path)
#         if not os.path.exists(parent_path.parent.absolute()):
#             os.makedirs(parent_path.parent.absolute())
#         plt.savefig(save_path, bbox_inches='tight', transparent=True)
#     return grid
    
#     grid.add_legend(title=hue.title(), bbox_to_anchor=(1, 0.87))
#     # for ax in grid.axes:
#     #     ax.tick_params(bottom=False)
#     if col is None:
#         grid.ax.tick_params(bottom=False)
#         for edge in range(df[x].nunique() * 3):
#             grid.ax.collections[edge].set_edgecolor('black')
#         for edge in range(df[x].nunique()):
#             grid.ax.get_children()[4 + (edge) * 5].set_color('black')
#             grid.ax.get_children()[5 + (edge) * 5].set_color('black')

#     else:
#         plt.subplots_adjust(hspace=0.2)
#         for subplot_title, ax in grid.axes_dict.items():
#             ax.set_title(subplot_title, pad=40)
#         for ax in grid.axes:
#             ax.title.set_position([.5, 2])
#             ax.tick_params(bottom=False)
#             for edge in range(df[x].nunique() * 3):
#                 ax.collections[edge].set_edgecolor('black')
#             for edge in range(df[x].nunique()):
#                 ax.get_children()[4 + (edge) * 5].set_color('black')
#                 ax.get_children()[5 + (edge) * 5].set_color('black')
#     return grid

