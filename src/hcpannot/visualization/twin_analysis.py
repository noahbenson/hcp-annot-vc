import os
import warnings
import numpy as np
import pandas as pd
import random
import pingouin as pg
from tqdm import tqdm
from itertools import product


def calculate_icc(long_twin_df, surface_area_df, twin_type, hemi, roi, icc_type):
    
    if hemi in ['lh', 'rh']:
        ratings = f'{hemi}_{roi}_percent'
    elif hemi == 'sum':
        ratings = f'{roi}_percent'
        surface_area_df[ratings] = surface_area_df[f'lh_{ratings}'] + surface_area_df[f'rh_{ratings}']
    tmp_cols = ['sid', ratings]
    tmp = long_twin_df.query('twin_type == @twin_type')
    tmp = tmp.merge(surface_area_df[tmp_cols], on='sid')
    icc = f'ICC{icc_type}'
    result = pg.intraclass_corr(data=tmp, 
                                targets='twin_index', 
                                raters='sid_type', 
                                ratings=ratings, 
                                nan_policy='omit')
    result = result.query('Type == @icc')
    result = result.drop(columns={'Description'})
    result['hemi'] = [hemi]
    result['ROI'] = [roi]
    result['twin_type'] = [twin_type] 
    return result

def calculate_icc_for_all(long_twin_df, surface_area_df, twin_types, hemis, rois, icc_type, save_path=None):
    
    if save_path is not None and os.path.exists(save_path):
        return pd.read_hdf(save_path)
    else:
        print(f'{save_path} doesn''t exist. calculating now...')
              
        twin_icc_df = pd.DataFrame({})
        for twin_type, hemi, roi in tqdm(product(twin_types, hemis, rois)):
            result = calculate_icc(long_twin_df, surface_area_df, twin_type, hemi, roi, icc_type)
            twin_icc_df = pd.concat((twin_icc_df, result), ignore_index=True)
    # Ignore PyTables PerformanceWarning
        if save_path is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=pd.io.pytables.PerformanceWarning)
                twin_icc_df.to_hdf(save_path, key='df')
            print(f'{save_path} saved')
        return twin_icc_df
    
def sample_with_condition(df, n_samples=100, min_count=10):
    """sample at least 5 pairs for each twin type while allowing resampling"""
    while True:
        sampled_numbers = random.choices(df.twin_index.unique(), k=n_samples)
        sampled_df = pd.DataFrame({})
        for new_i, i in enumerate(sampled_numbers):
            tmp = df.query('twin_index == @i')[['sid_type','twin_type','sid']]
            tmp['twin_index'] = new_i
            sampled_df = pd.concat((sampled_df, tmp), ignore_index=True)
        counts = sampled_df['twin_type'].value_counts()
        if all(counts >= min_count):
            break
    return sampled_df

def filter_nan_and_unrelated_pairs(twin_df, sampled_surface_area_df):
    """Filter out unrelated pairs & pairs that has nan values"""
    nan_list = sampled_surface_area_df[sampled_surface_area_df.isna().any(axis=1)]['sid'].tolist()
    filtered_df = twin_df.query('twin_type != "unrelated_pairs" & sid not in @nan_list')
    counts = filtered_df.twin_index.value_counts()
    # Step 2: Filter counts to get only those with more than one occurrence
    filtered_counts = counts[counts > 1]
    # Step 3: Filter the original DataFrame to keep only those rows
    # whose twin_index is in the filtered_counts index
    filtered_df = filtered_df[filtered_df['twin_index'].isin(filtered_counts.index)]
    return filtered_df

def bootstrap_calculate_icc_for_all(long_twin_df, surface_area_df, n_samples, n_bootstraps, hemi, roi, icc_type=2, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        return pd.read_hdf(save_path)
    else:
        np.random.seed(0) # For reproducible results
        sampled_surface_area_df = surface_area_df[['sid', f'lh_{roi}_percent',f'rh_{roi}_percent']]
        filtered_df = filter_nan_and_unrelated_pairs(long_twin_df, sampled_surface_area_df)
        twin_icc_df = pd.DataFrame({})
        for i in tqdm(range(n_bootstraps)):
            bts_tmp_df = sample_with_condition(filtered_df, n_samples, min_count=10)
            mono_result = calculate_icc(bts_tmp_df, surface_area_df, 'monozygotic_twins', hemi, roi, icc_type)
            mono_result = mono_result.rename(columns={'ICC': 'mono_ICC'})
            dizy_result = calculate_icc(bts_tmp_df, surface_area_df, 'dizygotic_twins', hemi, roi, icc_type)
            dizy_result = dizy_result.rename(columns={'ICC': 'dizy_ICC'})
            bts_df = pd.concat((mono_result, dizy_result['dizy_ICC']), axis=1)
            bts_df['bootstrap'] = i
            bts_df = bts_df[['bootstrap','ROI','hemi','mono_ICC','dizy_ICC']]
            twin_icc_df = pd.concat((twin_icc_df, bts_df), axis=0, ignore_index=True)  
        twin_icc_df['h2'] = 2*(twin_icc_df['mono_ICC'] - twin_icc_df['dizy_ICC']) #Falconer's formula
        # Ignore PyTables PerformanceWarning
        if save_path is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=pd.io.pytables.PerformanceWarning)
                twin_icc_df.to_hdf(save_path, key='df')
        return twin_icc_df
    
def calculate_confidential_interval(df, to_calculate, to_group=['ROI', 'hemi']):
    ci_68_df = df.groupby(to_group)[to_calculate].apply(lambda x: [np.percentile(x, 16), np.percentile(x, 84)])
    ci_68_df = ci_68_df.reset_index().rename(columns={to_calculate: f'{to_calculate}_ci_68'})
    ci_95_df = df.groupby(to_group)[to_calculate].apply(lambda x: [np.percentile(x, 2.5), np.percentile(x, 97.5)])
    ci_95_df = ci_95_df.reset_index().rename(columns={to_calculate: f'{to_calculate}_ci_95'})
    ci_df = pd.merge(ci_68_df, ci_95_df, on=to_group)
    return ci_df
