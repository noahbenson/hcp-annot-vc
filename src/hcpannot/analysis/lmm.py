from hcpannot.config import (ventral_raters, meanrater)
from hcpannot.proc import proc

import pimms
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
import neuropythy as ny

def postprocess_result(result, native_aligned=False):
    '''
    Postprocess the result of the linear mixed-effects model. The function sets the variance explained by researchers to 0
    and the variance explained by subjects to 1 for the first data point of the hV4_outer and VO_outer contours.
    
    The function also calculates the residual variance (variance not explained by researchers or subjects).
    '''
    
    for contour in ['hV4_outer', 'VO_outer']: 
        # the starting point of these two contours is the same for all researchers if fsaverage data is used
        # therefore, the variance explained by researchers is conceptually 0 at data point 0
        
        for hemi in ['lh', 'rh']:
            
            if not native_aligned: # if fsaverage data is used
                
                # check if the result for data point 0 of the contour exists
                exists = ((result['contour'] == contour) & (result['loc_on_contour'] == 0) & (result['hemi'] == hemi)).any()

                if exists: 
                    # if the result exists, set the variance explained by researchers to 0
                    result.loc[(result['contour'] == contour) & (result['loc_on_contour'] == 0) & 
                                   (result['hemi']==hemi), 'varex_rater'] = 0
                    # set the variance explained by subjects to 1
                    result.loc[(result['contour'] == contour) & (result['loc_on_contour'] == 0) & 
                                   (result['hemi']==hemi), 'varex_sbj'] = 1

                else:  
                    # if the result does not exist (e.g. model failed to converge), add a new row to the dataframe
                    # where the variance explained by researchers is 0 and the variance explained by subjects is 1
                    new_row = pd.DataFrame([[0, 1, hemi,contour, 1]], columns=['varex_rater', 'varex_sbj', 
                                                                                'hemi', 'contour', 'loc_on_contour'])
                    # append the new row to the dataframe
                    result=pd.concat([result,new_row], ignore_index=True)
    
    # sort the dataframe by contour, hemisphere, and location on the contour
    result = result.sort_values(by=['contour', 'hemi', 'loc_on_contour'], ignore_index=True)
    
    # calculate the residual variance (variance not explained by researchers or subjects)
    result['residual'] = 1 - result['varex_rater'] - result['varex_sbj']
    
    return result

def get_contour_data (result, contour): # get the data for a specific contour
    
    '''
    Get the data for a specific contour. The function returns a dictionary that contains the variance explained by researchers,
    the variance explained by subjects, and the residual variance for the left and right hemisphere.
    '''

    # create a dictionary to store the data
    contour_data = defaultdict(dict)
    
    # get the contour data
    lh_dat = result[(result['hemi'] == 'lh') & (result['contour'] == contour)]
    rh_dat = result[(result['hemi'] == 'rh') & (result['contour'] == contour)]

    # get the variance explained by subjects and store the data in the dictionary
    contour_data['lh']['sbj'] = lh_dat['varex_sbj'].values
    contour_data['rh']['sbj'] = rh_dat['varex_sbj'].values

    # get the variance explained by researchers and store the data in the dictionary
    contour_data['lh']['rater'] = lh_dat['varex_rater'].values
    contour_data['rh']['rater'] = rh_dat['varex_rater'].values
    
    # get the residual of x coordinates and store the data in the dictionary
    contour_data['lh']['residual'] = lh_dat['residual'].values
    contour_data['rh']['residual'] = rh_dat['residual'].values
    
    # check if r2s_variance was a column in the dataframe
    if 'r2s_variance' in result.columns:
        # get the variance of the random slopes and store the data in the dictionary
        contour_data['lh']['r2s_variance'] = lh_dat['r2s_variance'].values
        contour_data['rh']['r2s_variance'] = rh_dat['r2s_variance'].values
    
    return contour_data

def lwplot(x, y, axes=None, fill=True, edgecolor=None, color=None, **kw):
    '''
    lwplot(x, y) is equivalent to pyplot.plot(x, y), however the linewidth or lw options
      are interpreted in terms of the the coordinate system instead of printer points.
    lwplot(x, y, ax) plots on the given axes ax.
    
    All optional arguments that can be passed to pyplot's Polygon can be passed to lwplot.
    '''
    from neuropythy.util import zinv
    lw = kw['linewidth'] if 'linewidth' in kw else kw['lw'] if 'lw' in kw else None
    if 'linewidth' in kw: lw = kw.pop('linewidth')
    elif 'lw' in kw: lw = kw.pop('lw')
    else: lw = 0
    axes = plt.gca() if axes is None else axes
    if len(x) < 2: raise ValueError('lwplot line must be at least 2 points long')
    # we plot a particular thickness; we need to know the orthogonals at each point...
    pts = np.transpose([x,y])
    dd  = np.vstack([[pts[1] - pts[0]], pts[2:] - pts[:-2], [pts[-1] - pts[-2]]])
    nrm = np.sqrt(np.sum(dd**2, 1))
    dd  *= np.reshape(zinv(nrm), (-1,1))
    dd  = np.transpose([dd[:,1], -dd[:,0]])
    # we make a polygon or a trimesh...
    if pimms.is_vector(lw): lw = np.reshape(lw, (-1,1))
    xy = np.vstack([pts + lw*dd, np.flipud(pts - lw*dd)])
    n = len(pts)
    if pimms.is_vector(color) and len(color) == n:
        clr = np.concatenate([color, np.flip(color)])
        (nf0,nf1) = (np.arange(n-1), np.arange(1,n))
        (nb0,nb1) = (2*n - nf0 - 1,  2*n - nf1 - 1)
        tris = np.hstack([(nf0, nf1, nb0), (nb0, nb1, nf1)]).T
        (x,y) = xy.T
        tri = mpl.tri.Triangulation(x, y, tris)
        if 'cmap' not in kw: kw['cmap'] = 'hot'
        return axes.tripcolor(tri, clr, shading='gouraud',
                              linewidth=0, **kw)
    else:
        pg  = plt.Polygon(xy, closed=True, fill=fill, edgecolor=edgecolor,
                          linestyle=None, linewidth=0, color=color, **kw)
        return axes.add_patch(pg)
    
def plot_hmap(x, y, hemi, contour, lm_result, ax, cmap='hot', residual=False, max_var=None):
    
    '''Plot the heatmap of the variance explained by researchers to 
    '''
    
    # get the data for the contour
    data = get_contour_data(lm_result, contour)
        
    if not residual: # if not plotting the residual
        
        # get the ratio of variance explained by researchers to variance explained by subjects
        var_r2s = data[hemi]['r2s_variance']
        
        # the maximum ratio of variances is needed to set the maximum value of the heatmap
        # if not provided, the maximum value is set to the maximum of all values in the dataset
        # which includes data of both hemispheres, so that the colorbar is consistent across hemispheres
        if max_var == None: 
            max_var = max(lm_result['r2s_variance'])

        # plot the heatmap
        hmap = lwplot(x, y, axes=ax, cmap=cmap, vmin=0, vmax=max_var, color=var_r2s, lw=1+var_r2s*5)

    else: # if plotting the residual
        
        # get the residual
        residual = data['residual']
        hmap = lwplot(x, y, axes=ax, cmap=cmap, vmin=0, vmax=max(residual), color=residual, lw=0.5+residual*2)

    return hmap


def plot_lw (result, hemi, sub_contours, contour_save_path, proc_path, flatmap=True, ax=None):
    
    npoints = 50
    meanlines = ny.load('/data/crcns2021/hcpannot-cache/annot-v123/999999.json.gz')
    
    if ax is None:
        ax = plt.gca()
     
    if flatmap: # plot flatmap
        
        # plot mean contour for v123
        meanv123 = meanlines[hemi]
        for t in meanv123.keys():
            for c in meanv123[t]:
                trace = meanv123[t][c].curve.linspace(npoints)
                ax.plot(trace[0], trace[1], 'w-')
       
        fsaverage_hem = ny.freesurfer_subject('fsaverage').hemis[hemi]
        flatmap = ny.to_flatmap('occipital_pole', fsaverage_hem, radius=np.pi/2)
        ny.cortex_plot(flatmap, axes=ax)
        
        # define line width and style for each contour
        lw = 0.3
        style = 'g-'

    else:
        # define line width and style for each contour
        lw = 1
        style = 'k-'
        
        # also plot V3v
        sub_contours.append('V3v')
        
    # get mean ventral contour
    mean = proc('ventral', rater='mean', sid=999999, hemisphere=hemi, load_path=contour_save_path, save_path=proc_path)
    mean_traces = mean['traces']
    

    # plot individual ventral contour
    for contour in sub_contours:
        
        x_coords, y_coords = mean_traces[contour].curve.linspace(npoints)
        
        if not flatmap and hemi == 'rh':
            x_coords += 75
            
        ax.plot(x_coords, y_coords, style, alpha=0.5, lw=lw)
            
        if contour == 'V3v':
            continue
        else:
            hmap = plot_hmap(x=x_coords, 
                            y=y_coords, 
                            hemi=hemi, 
                            contour=contour, 
                            lm_result=result, 
                            max_var=max(result['r2s_variance']),
                            ax=ax)

    return hmap
    
    