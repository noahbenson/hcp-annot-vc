import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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


def get_correlation_matrix(df):
    corr_matrix = df.corr()
    mask = np.triu(corr_matrix)
    return corr_matrix, mask


def heatmap_surface_area(df, mask=None, height=7, cmap="YlOrRd",
                         annot=True, boundary_line=None,
                         fmt=".2f", vmin=0, vmax=1):
    sns.set(style={'axes.facecolor': 'white', 'font.family': 'Helvetica'},
            rc={'axes.labelpad': 20, 'figure.figsize': (height, height)},
            font_scale=height / 6)
    if annot is True:
        annot_kws = {"size": 35 / np.sqrt(len(df))}
    else:
        annot_kws = None
    ax = sns.heatmap(df, mask=mask,
                       annot=annot, annot_kws=annot_kws, fmt=fmt,
                       cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .7},
                       linewidth=.3, square=True)
    if boundary_line is not None:
        ax.hlines(boundary_line, ax.get_xlim()[0], (ax.get_xlim()[1]/2)+0.5, color='blue', linewidth=2, linestyles='--'),
        ax.vlines(boundary_line, (ax.get_ylim()[0]/2)+0.5, ax.get_ylim()[0], color='blue', linewidth=2, linestyles='--')
    return ax

def violinplot_surface_area(df, x, x_order, hue='hemisphere', hue_order=['lh','rh'], split=True,
                            col=None, col_wrap=None,
                            relative_area=True, height=8, cmap=sns.color_palette("Spectral")):
    sns.set(style={'axes.facecolor':'white', 'font.family':'Helvetica'},
            rc={'axes.labelpad': 25}, font_scale=height/3)
    sns.despine(top=True, bottom=True, right=True)
    if relative_area is True:
        df['p_surface_area'] = df.apply(lambda x: calculate_percent(x['surface_area'], cortex=x['cortex']), axis=1)
        y = 'p_surface_area'
        y_label = 'Relative surface area (%)'
    else:
        y = 'surface_area'
        y_label = r'Surface area ($mm^2$)'
    grid = sns.FacetGrid(df,
                         col=col, col_wrap=col_wrap,
                         height=height,
                         aspect=1.3,
                         legend_out=True,
                         sharex=True, sharey=True)
    grid = grid.map(sns.violinplot, x, y, hue,
                    hue_order=hue_order, split=split, order=x_order, palette=cmap, cut=0,
                    inner='box', linewidth=2, saturation=0.9, bw=.15, edgecolor='black')
    grid.set_axis_labels('', y_label)
    grid.add_legend(title=hue.title(), bbox_to_anchor=(1, 0.87))
    if col is None:
        for edge in range(df[x].nunique() * 3):
            grid.ax.collections[edge].set_edgecolor('black')
        for edge in range(df[x].nunique()):
            grid.ax.get_children()[4 + (edge) * 5].set_color('black')
            grid.ax.get_children()[5 + (edge) * 5].set_color('black')
    else:
        plt.subplots_adjust(hspace=0.2)
        for subplot_title, ax in grid.axes_dict.items():
            ax.set_title(subplot_title, pad=40)
        for ax in grid.axes:
            ax.title.set_position([.5, 2])
            for edge in range(df[x].nunique() * 3):
                ax.collections[edge].set_edgecolor('black')
            for edge in range(df[x].nunique()):
                ax.get_children()[4 + (edge) * 5].set_color('black')
                ax.get_children()[5 + (edge) * 5].set_color('black')
    return grid

