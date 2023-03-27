import os
import sys
import pandas as pd
import seaborn as sns

def calculate_percent(roi, cortex):
    return roi*100/cortex

def melt_rois_to_lh_rh(df, roi, id_vars=['sid','anatomist']):
    original_roi_cols = [col for col in df if roi in col]
    df = df[original_roi_cols + id_vars]
    hemi_cols = [col[3].lower()+'h' for col in original_roi_cols]
    df = df.rename(columns=dict(zip(original_roi_cols, hemi_cols)))
    df = pd.melt(df, id_vars=id_vars, value_vars=hemi_cols, var_name='hemisphere', value_name=roi, ignore_index=True)
    return df

def violinplot_surface_area(df, hue='hemisphere', percentage=True):
    if percentage is True:
        y = 'p_surface_area'
    else:
        y = 'surface_area'
    sns.set(style={'axes.facecolor':'white', 'font.family':'Helvetica'}, rc={'figure.figsize':(8,5), 'axes.labelpad': 20}, font_scale=1.5)
    grid = sns.violinplot(data=df, x="roi", y=y,
                          order=['hV4', 'VO1', 'VO2'], cut=0,
                          hue='hemisphere', split=(hue!=None),
                          orient="v", linewidth=2, dodge=True, bw=.15, palette='Reds', height=10)
    grid.set_ylabel(ylabel="Surface Area (%)", labelpad=25)
    grid.set_xlabel(xlabel="", labelpad=25)
    grid.tick_params(bottom=False)
    sns.despine(top=True, bottom=True, right=True)
    lgd = grid.legend(title='Hemisphere', loc='upper left', bbox_to_anchor=(0.99, 1.05))
    lgd.get_frame().set_edgecolor('none')
    return grid