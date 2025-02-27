"""Averaging contours of the same subject among researchers and plotting the contours on flatmaps.
"""

from hcpannot.proc import (proc, rigid_align_points)
from hcpannot.config import (ventral_raters, meanrater)
import neuropythy as ny
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# list of raters
ventral_raters = ventral_raters + [meanrater]

# generate a list of colors
colors = ['r', 'g', 'b', 'c', 'm', 'k']
rater_colors = {r: c for r, c in zip(ventral_raters, colors)}

def plot_rater_contours(raters, subject_id, hemi, contours, 
                  save_path, data_path, ax=None, lw=None):
    """Plot contours of the same subject from different raters.
    """
    
    # check if raters and contours are lists
    # if not, convert them to lists
    if not isinstance(raters, (tuple, list, np.ndarray)):
        raters = [raters]
        
    if not isinstance(contours, (tuple, list, np.ndarray)):
        contours = [contours]
    
    # if no axis is provided, use the current axis    
    if ax is None:
        ax = plt.gca()
    
    # to avoid duplicate legend entries
    raters_legend = set() 
    
    for r in raters:
        
        # get the line color based on the rater
        color = rater_colors.get(r, 'gray')
        
        for c in contours:
            
            # get the coordinates of the contours
            try:
                dat = proc('ventral', rater=r, sid=subject_id, hemisphere=hemi, 
                           save_path=save_path, load_path=data_path)
                
                coords = dat['fsaverage_traces'][c].points

            except Exception as e:
                print(f'Error processing contours: {e}')
            
            # plot the contours 
            
            # if the rater is not in the legend yet, add them to the legend
            if r not in raters_legend:
                raters_legend.add(r)
                ax.plot(coords[0], coords[1], color=color, lw=lw, label=f'{r}')
            
            # otherwise, plot without the label
            else:
                ax.plot(coords[0], coords[1], color=color, lw=lw)    
    
    ax.legend()
            
    return

def create_LineCollection(sids, hemi, roi, rater, 
                          lw, alpha, save_path,
                          data_path, color=None):
    
    """Create a LineCollection object for contours of the same ROI from the same
    researcher from different subjects to be plotted together effectively
    """
    
    lines = [] # list of coordinates for each subject
    
    # for each subject, try to load the contour coordinates
    for sid in sids:
        
        try:
            dat = proc(contours_plan='ventral', rater=rater, sid=sid, hemisphere=hemi, save_path=save_path, load_path=data_path)
            coords = dat['fsaverage_traces'][roi].points
            lines.append(np.column_stack(coords))
            
        except Exception as e:
            print(f'Error processing contours: {e}')
            continue
        
        # add V1-V3 contours to the list of lines

    lc = LineCollection(lines, linewidths=lw, alpha=alpha, colors=color)
    return lc

def plot_lc(sids, hemi, rois, save_path, data_path,
            rater=meanrater, ax=None, lw=0.25, 
            alpha=0.1, colors=None):
    
    """Plot a LineCollection object on a flatmap. By default, the contours are plotted on
    flatmap with the mean V1-V3 contours.
    """
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=(3.5,3.5), dpi=1200)
    
    if not isinstance(rois, list):
        rois = [rois]
        
    # if plot_v123:
    #     if meanlines is None or 'lh' not in meanlines.keys() or 'rh' not in meanlines.keys():
    #         raise ValueError('Mean V1-V3 contours are not provided.')
    #     else:
    #         meanv123 = meanlines[hemi]
    #         for t in meanv123.keys():
    #             for c in meanv123[t]:
    #                 trace = meanv123[t][c].curve.linspace(npoints)
                    
    #                 if hemi == 'lh':
    #                     ax.plot(trace[0]+8, trace[1], 'k-') # shift the x coordinates by 8
    #                 else:
    #                     ax.plot(trace[0]-8, trace[1], 'k-') # shift the x coordinates by -8
    
    for roi in rois:
        
        lc = create_LineCollection(sids, hemi, roi, rater,
                                    lw=lw, alpha=alpha, color=colors.get(roi, None),
                                    save_path=save_path, data_path=data_path) 

        ax.add_collection(lc)
        
    ax.set_title('Ventral Contours')
    ax.set_ylim(-75, 40)
    ax.axis('equal')
    ax.axis('off')

    return

def align_fsnative(sid, hemi, contours, rater, proc_path, npoints=500):
    
    """For a given subject in a given hemisphere delineated by a given rater,
    align the contours to mean contours in the fsaverage space.
    """
    
    mean_coords = []

    # TODO: check if order of contours matters
    
    for contour in contours:
        
        # define cache file path
        mean_cache_file = os.path.join(proc_path, 'fsaverage', f'cacherater_{meanrater}_{sid}_{hemi}_{contour}.mgz')
        
        # check if the mean contour coordinates are already saved
        if os.path.isfile(mean_cache_file):
            # if the coordinates are already saved, load them
            # and append them to the list of mean coordinates
            coords = ny.load(mean_cache_file)
            mean_coords.extend(coords)
        else:
            print(f'Mean contour coordinates for {sid} {hemi} {contour} not found. Skipping...')
            return
            
    # align the contours to the mean contours in fsaverage space
    # and save the aligned coordinates

    rater_coords = []
    
    # define cache file path for the rater
    for contour in contours:
        cache_file = os.path.join(proc_path, 'fsnative', f'cacherater_{rater}_{sid}_{hemi}_{contour}.mgz')
        
        if os.path.isfile(cache_file):
            coords = ny.load(cache_file)
            rater_coords.extend(coords)
        else:
            print(f'Contour coordinates for {rater} {sid} {hemi} {contour} not found. Skipping...')
            return
        
    # try to align the rater contours to the mean contours
    # TODO: inpsect the aligned contours
    try:
        aligned_contours = rigid_align_points(rater_coords, mean_coords)
    except Exception as e:
        print(f'Error aligning contours: {e}')
        return
    
    # break the aligned contours into separate contours
    for i, contour in enumerate(contours):
        aligned_contour = aligned_contours[:, npoints*i:npoints*(i+1)]
        
        # save the aligned contours
        file = os.path.join(proc_path, 'fsnative_aligned', f'cacherater_{rater}_{sid}_{hemi}_{contour}.mgz')
        ny.save(file, aligned_contour)
        
    return