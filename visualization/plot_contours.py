import os
import numpy as np
import neuropythy as ny
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
from glob import glob
from hcpannot.proc import proc
from hcpannot.config import meanrater
from hcpannot import (save_contours, load_contours)
import seaborn as sns
from pathlib import Path
from visualization import utils
import matplotlib as mpl

font_rc = {'family':'sans-serif',
           'sans-serif':['HelveticaNeue', 'Helvetica', 'Arial'],
          'weight': 'light'}
mpl.rc('font', **font_rc)

def set_rcParams(rc):
    if rc is None:
        pass
    else:
        plt.rcParams.update(rc)

def set_fontsize(small, medium, large):
    font_rc = {'font.size': medium,
          'axes.titlesize': large,
          'axes.labelsize': medium,
          'xtick.labelsize': small,
          'ytick.labelsize': small,
          'legend.fontsize': small,
          'figure.titlesize': large}
    set_rcParams(font_rc)


def load_cortex(subject, hemi):
    cortex = ny.data['hcp_lines'].subjects[subject].hemis[hemi]  # cortex object
    return cortex


def roi_trace_to_path(dat, cortex, roi):
    # grab a trace
    trace = dat['traces'][roi]
    # get the subject's hemisphere mesh
    path = trace.to_path(cortex)
    return path



def normalize_coords(fsa_coords, n_points=500):
    curve = ny.util.CurveSpline(fsa_coords)
    normed_fsa_coords = curve.linspace(n_points)
    return normed_fsa_coords


def arrange_multiple_coords_into_dict(normed_x_list,
                                      normed_y_list,
                                      normed_fsa_coords_keys=None,
                                      average=True):
    if normed_fsa_coords_keys is None:
        normed_fsa_coords_keys = np.range(0, len(normed_x_list))
    normed_fsa_coords_vals = [np.asarray((x, y)) for x, y in zip(normed_x_list, normed_y_list)]
    normed_fsa_coords_dict = dict(zip(normed_fsa_coords_keys, normed_fsa_coords_vals))
    if average is True:
        avg_x = np.mean(normed_x_list, axis=0)
        avg_y = np.mean(normed_y_list, axis=0)
        normed_fsa_coords_dict['avg'] = np.asarray((avg_x, avg_y))
    return normed_fsa_coords_dict


def _display_msg(message, verbose):
    if verbose:
        print(message)
    else:
        pass



def get_trace_coords_in_fsaverage(coords, annot):
    cortex = annot['cortex']
    fmap1 = annot['flatmap']
    fmap2 = ny.to_flatmap('occipital_pole', cortex)
    addr = fmap1.address(coords)
    fsa_coords = fmap2.unaddress(addr)
    return fsa_coords

def make_fsaverage_contours(stream,
                           rater,
                           hemisphere,
                           roi,
                           sid,
                           n_points,
                           data_dir,
                           proc_dir,
                           cache_file,
                           verbose=True,
                           return_annot=False):
    """ 
    data_dir is load_path of proc(). It should be a directory containing contours.json files.
    save_path is a path including .mgz file name for a matrix converted to a matrix. 
    The parent directory of save_path will be used as a save_path argument for the proc() (=proc_dir) """
    _display_msg(f'......subject no.{sid}', verbose)
    cache_path = os.path.join(proc_dir, cache_file)
    if os.path.isfile(cache_path):
        _display_msg(f'found cache data!', verbose)
        fsa_coords = ny.load(cache_path)
    else:
        _display_msg(f'Start making coordinates...', verbose)
        os.makedirs(proc_dir, mode=0o775, exist_ok=True)
#         if rater == 'mean':
#             _display_msg(f'No cache data found. Using meanproc()....', verbose)
#             annot = meanproc(stream,
#                              load_path=data_dir,
#                              sid=sid,
#                              hemisphere=hemisphere,
#                              save_path=proc_dir)
#         else:
#             _display_msg(f'No cache data found. Using proc()....', verbose)
        annot = proc(stream,
                     rater=rater,
                     load_path=data_dir,
                     sid=sid,
                     hemisphere=hemisphere,
                     save_path=proc_dir)

        trace = annot['traces'][roi]
        coords = trace.curve.linspace(n_points)
        fsa_coords = get_trace_coords_in_fsaverage(coords, annot)
        parent_path = Path(cache_path)
        os.makedirs(parent_path.parent.absolute(), exist_ok = True)
        ny.save(cache_path, fsa_coords)    
      
    return fsa_coords


def main(rater,
         hemi,
         roi,
         subject_list,
         n_points=800,
         verbose=False,
         cache_file_list=None,
         trace_save_path=None,
         average=True):
    x_list, y_list = [], []
    _display_msg(f'**number of subjects: {len(subject_list)}\n**nuber of points:{n_points}', verbose)

    for sid, cache_file in tqdm(zip(subject_list, cache_file_list)):
        x, y = make_path(rater, hemi, roi, sid, n_points, verbose, cache_file, trace_save_path)
        x_list.append(x)
        y_list.append(y)
    normed_fsa_coords_dict = arrange_multiple_coords_into_dict(x_list, y_list, subject_list, average)
    return normed_fsa_coords_dict

def plot_multiple_path(normed_fsa_coords_dict, legend=True, average_only=False, **kwargs):
    title = kwargs.pop('title', None)
    if average_only:
        normed_fsa_coords_dict = {'avg': normed_fsa_coords_dict['avg']}
    n_lines = len(normed_fsa_coords_dict.keys())
    fig, ax = plt.subplots(1,1)
    pal = plt.cm.gist_rainbow(np.linspace(0, 1, n_lines))
    color = iter(pal)
    for k, fsa_coords in normed_fsa_coords_dict.items():
        (x,y) = fsa_coords
        c=next(color)
        if k == 'avg':
            c = 'k'
        ax.plot(x,y, label=str(k), color=c, **kwargs)
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncols=2)
    ax.get_legend().set_visible(legend)
    plt.tight_layout()

def get_contour_list_drawn_by_rater(trace_save_path, hemi, roi, rater, return_full_path=False):

    path_list = glob(os.path.join(trace_save_path, rater, f'*/{hemi}.{roi}.json'))
    if return_full_path is True:
        return path_list
    else:
        sids_list = [int(k.split('/')[-2]) for k in path_list]
        return sids_list


def lwplot(x, y, axes=None, fill=True, edgecolor=None, color=None, **kw):
    '''
    lwplot(x, y) is equivalent to pyplot.plot(x, y), however the linewidth or lw options
      are interpreted in terms of the coordinate system instead of printer points.
    lwplot(x, y, ax) plots on the given axes ax.

    All optional arguments that can be passed to pyplot's Polygon can be passed to lwplot.
    '''
    from neuropythy.util import zinv
    import pimms
    import matplotlib as mpl

    lw = kw['linewidth'] if 'linewidth' in kw else kw['lw'] if 'lw' in kw else None
    if 'linewidth' in kw:
        lw = kw.pop('linewidth')
    elif 'lw' in kw:
        lw = kw.pop('lw')
    else:
        lw = 0
    axes = plt.gca() if axes is None else axes
    if len(x) < 2: raise ValueError('lwplot line must be at least 2 points long')
    # we plot a particular thickness; we need to know the orthogonals at each point...
    pts = np.transpose([x, y])
    dd = np.vstack([[pts[1] - pts[0]], pts[2:] - pts[:-2], [pts[-1] - pts[-2]]])
    nrm = np.sqrt(np.sum(dd ** 2, 1))
    dd *= np.reshape(zinv(nrm), (-1, 1))
    dd = np.transpose([dd[:, 1], -dd[:, 0]])
    # we make a polygon or a trimesh...
    if pimms.is_vector(lw): lw = np.reshape(lw, (-1, 1))
    xy = np.vstack([pts + lw * dd, np.flipud(pts - lw * dd)])
    n = len(pts)
    if pimms.is_vector(color) and len(color) == n:
        clr = np.concatenate([color, np.flip(color)])
        (nf0, nf1) = (np.arange(n - 1), np.arange(1, n))
        (nb0, nb1) = (2 * n - nf0 - 1, 2 * n - nf1 - 1)
        tris = np.hstack([(nf0, nf1, nb0), (nb0, nb1, nf1)]).T
        (x, y) = xy.T
        tri = mpl.tri.Triangulation(x, y, tris)
        if 'cmap' not in kw: kw['cmap'] = 'hot'
        return axes.tripcolor(tri, clr, shading='gouraud',
                              linewidt=0, **kw)
    else:
        mean_contours = axes.plot(x, y, 'k-', linewidth=0)

        pg = plt.Polygon(xy, True, fill=fill, edgecolor=edgecolor,
                         linestyle=None, linewidth=0, color=color, **kw)

        return axes.add_patch(pg)

def find_interquartile_range(my_list, axis=0):
    """ takes in a 2d list which consists of subjects on the row and each dot point as a column.
    The default purpose is to find the IQR around each point index (across sid)"""
    return np.percentile(my_list, 75, axis=axis) - np.percentile(my_list, 25, axis=axis)

def calculate_IQR_summary(x, y, axis=0):
    rc = {'xtick.bottom': True,
          'xtick.labelbottom': True,
          'xtick.major.size': 2,
          'xtick.major.width': 2,
          'xtick.labelsize': 8,
          'ytick.left': True,
          'ytick.labelleft': True,
          'ytick.major.size': 2,
          'ytick.major.width': 2,
          'ytick.labelsize': 8,
          'axes.linewidth': 2,
          'axes.titlepad': 10,
          'axes.labelpad': 20,
          'figure.figsize':(7, 7*1.2)
          }
    x_iqr = find_interquartile_range(x, axis=axis)
    y_iqr = find_interquartile_range(y, axis=axis)
    return np.sqrt(x_iqr ** 2 + y_iqr ** 2)


def check_sids(df, sid, y=None, hue=None, hue_order=None,
               row=None, row_order=None,
               col=None, col_order=None,
               suptitle=' ',
               save_path=None, **kwargs):
    
    rc = {'xtick.bottom': True,
          'xtick.labelbottom': True,
          'figure.figsize':(7, 7*1.2)
          }    
    
    sns.set_context("notebook", font_scale=2)
    #sns.set_theme(style="ticks", context="notebook", font_scale=2)
    set_rcParams(rc)
    subj_ids = list(ny.data['hcp_lines'].subject_list)
    subj_ids.sort()
    df = df.replace({sid: dict(zip([999999]+subj_ids, np.arange(0, 182)))}) #999999 represents mean across subjects 
    grid = sns.FacetGrid(df,
                         col=col, col_order=col_order,
                         row=row, row_order=row_order,
                         aspect=3.3,
                         legend_out=True,
                         sharex=True, sharey=True, **kwargs)
    grid = grid.map_dataframe(sns.histplot, sid, y,
                              palette=sns.color_palette('hls', df[hue].nunique()),
                              hue=hue, hue_order=hue_order,
                              edgecolor='grey', linewidth=0.01,
                              multiple="stack", discrete=True)
    # for subplot_title, ax in grid.axes_dict.items():
    #     ax.set_title(f" ")
    grid.add_legend()
    grid.fig.suptitle(suptitle, fontweight='bold', y=1.05)
    grid.set_axis_labels('HCP subjects', 'Researchers')
    grid.set(xlim=(0, 182), xticks=(0, 90, 181))
    utils.save_fig(save_path)
    return grid