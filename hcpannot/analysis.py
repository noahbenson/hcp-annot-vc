################################################################################
# analysis.py
#
# Analysis tools for the HCP visual cortex contours.
# by Noah C. Benson <nben@uw.edu>

# Import things
import sys, os, pimms, json
import numpy as np
import pyrsistent as pyr
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import neuropythy as ny

from .core import (image_order, op_flatmap)
from .interface import (default_load_path, imgrid_to_flatmap, flatmap_to_imgrid,
                        default_imshape, subject_ids, subject_data)

# Global Variables #############################################################
roi_image_shape = default_imshape[0] // 2
# Subjects
subject_ids = np.array(subject_ids)
# Here we have the subject lists in the order we assigned them to the
# project's raters.
subject_list_1 = np.array(
    [100610, 102311, 102816, 104416, 105923, 108323, 109123, 111312,
    111514, 114823, 115017, 115825, 116726, 118225, 125525, 126426,
    128935, 130114, 130518, 131217, 132118, 145834, 146735, 157336,
    158136, 164131, 167036, 169747, 173334, 175237, 182436, 192439,
    198653, 201515, 203418, 214019, 221319, 318637, 320826, 346137,
    360030, 365343, 385046, 393247, 401422, 406836, 467351, 525541,
    573249, 581450, 627549, 644246, 671855, 690152, 732243, 783462,
    814649, 878776, 898176, 958976])
subject_list_2 = np.array(
    [134627, 140117, 146129, 148133, 155938, 158035, 159239, 164636,
    165436, 167440, 169040, 169343, 171633, 176542, 177140, 178647,
    181636, 182739, 187345, 191336, 191841, 192641, 195041, 199655,
    204521, 205220, 212419, 233326, 239136, 246133, 251833, 263436,
    283543, 389357, 395756, 429040, 436845, 541943, 550439, 552241,
    601127, 638049, 724446, 751550, 757764, 765864, 770352, 782561,
    818859, 825048, 859671, 871762, 878877, 899885, 910241, 927359,
    942658, 951457, 971160, 973770])
subject_list_3 = ~(np.isin(subject_ids, subject_list_1) |
                   np.isin(subject_ids, subject_list_2))
subject_list_3 = np.sort(subject_ids[subject_list_3])

# Data about the visual cortex contours we are analyzing.
vc_contours = {'hV4_VO1': '{hemisphere}.hV4_VO1.json',
               'VO1_VO2': '{hemisphere}.VO1_VO2.json',
               'hV4_outer': '{hemisphere}.hV4.json',
               'VO_outer': '{hemisphere}.VO_outer.json'}
vc_contours_meanrater = {'hV4_VO1': '{hemisphere}.hV4_VO1.json',
                         'VO1_VO2': '{hemisphere}.VO1_VO2.json',
                         'outer': '{hemisphere}.outer.json'}
# A tuple of all the traces currently generated.
all_traces = ('hV4','VO1','VO2','outer','hV4_VO1','VO1_VO2')
# The name of the rater used as mean.
meanrater = 'mean'


# Plan Functions ###############################################################
@pimms.calc('chirality')
def parse_chirality(hemisphere):
    """Parses the chirality (`'lh'` or `'rh'`) from a hemisphere name.
    
    Parameters
    ----------
    hemisphere : str
        The name of a hemisphere to use in the calculations. This should be the
        name of a HCP subject hemisphere such as `'lh'` (LH native) or 
        `'rh_LR59k_MSMAll'` (RH 59k fs_LR hemisphere with MSMAll alignment).
        Typically, this should be either `'lh'` or `'rh'`.
        
    Outputs
    -------
    chirality : str
        Either `'lh'` or `'rh'`, depending on whether the given hemisphere is a
        left or right hemisphere, respectively.
    """
    import neuropythy as ny
    return ny.to_hemi_str(hemisphere.split('_')[0])
def to_data_path(rater, sid, save_path, mkdir=False, mkdir_mode=0o775):
    """Returns a save path for the given rater, subject ID, and save path.

    `to_data_path(rater, sid, save_path)` appends directories for the rater and
    subject id (`sid`) to the given `save_path` and returns it. This is roughly
    equivalent to `os.path.join(save_path, rater, str(sid))`.

    In addition to joining the path, `to_data_path` expands variables and user
    components of the path, and the `mkdir` and `mkdir_mode` options may be
    passed to ensure that the directories are created.
    """
    data_path = os.path.join(save_path, rater, str(sid))
    data_path = os.path.expanduser(os.path.expandvars(data_path))
    if mkdir and not os.path.isdir(data_path):
        os.makedirs(data_path, mode=mkdir_mode)
    return data_path
def load_contours(rater, sid, h, save_path,
                  vc_contours=vc_contours):
    """Loads a set of contours for a subject and hemisphere from a save path.

    `load_contours(rater, sid, h, save_path)` loads a set of contours from the
    given `save_path` for the given rater, subject ID (`sid`) and hemisphere
    (`h`). These contours are returned as a dict whose keys are the contours
    names and whose values are the contour points.

    The optional argument `vc_contours` can be passed to specify that specific
    contours be loaded.
    """
    import os, neuropythy as ny, pyrsistent as pyr
    data_path = to_data_path(rater, sid, save_path=save_path)
    h = ny.to_hemi_str(h)
    # We need to load precisely these files:
    contours = {}
    for (name,fname) in vc_contours.items():
        fname = fname.format(hemisphere=h, sid=sid, rater=rater)
        fname = os.path.join(data_path, fname)
        if not os.path.isfile(fname):
            raise RuntimeError(("contour file missing for "
                                f"{rater}/{sid}/{h}/{name}: {fname}"))
        with open(fname, 'r') as file:
            c = np.array(json.load(file))
        # Convert c into flatmap contours.
        contours[name] = imgrid_to_flatmap(c.T)
    return contours
def save_contours(rater, sid, h, contours, save_path,
                  vc_contours=vc_contours,
                  overwrite=True, mkdir=True, mkdir_mode=0o775):
    """Saves a set of contours for a subject and hemisphere to a save path.

    `save_contours(rater, sid, h, contours, save_path)` saves a set of contours
    (`contours`) to the given `save_path` for the given rater, subject ID
    (`sid`) and hemisphere (`h`). The return value is a dict whose keys are the
    contour names and whose values are the filenames to which the contour was
    saved.

    The optional argument `vc_contours` can be passed to specify that specific
    contours be saved.
    """
    import os, neuropythy as ny, pyrsistent as pyr
    data_path = to_data_path(rater, sid, save_path=save_path)
    if mkdir and not os.path.isdir(data_path):
        os.makedirs(data_path, mode=mkdir_mode)
    h = ny.to_hemi_str(h)
    fls = {}
    for (name,fname) in vc_contours.items():
        cnt = contours.get(name)
        if cnt is None: continue
        elif ny.is_path_trace(cnt): cnt = cnt.points
        cnt = np.asarray(cnt)
        fname = fname.format(hemisphere=h, sid=sid, rater=rater)
        fname = os.path.join(data_path, fname)
        if overwrite or not os.path.isfile(fname):
            # Convert c into flatmap contours.
            cnt = flatmap_to_imgrid(cnt)[0,0].tolist()
            with open(fname, 'w') as file:
                json.dump(cnt, file)
            fls[name] = fname
    return fls
@pimms.calc('data_path', 'raw_contours')
def calc_load_contours(rater, sid, chirality, save_path, vc_contours=vc_contours):
    """Load the contours for a rater, subject, and hemisphere.
    
    Parameters
    ----------
    rater : str
        The rater whose contours are to be loaded.
    sid : int
        The HCP subject-ID of the subject to load.
    chirality : 'lh' or 'rh'
        The hemisphere (`'lh'` or `'rh'`) to load.
    save_path : str
        The path of the save directory from the `hcp-annot-vc:data` respotiroy.
    vc_contours : dict, optional
        A dictionary whose keys are the names of the contours and whose values
        are the filenames. The default value for this option is correct unless
        you are running the calculation in an context other than the one it was
        written for.

    Outputs
    -------
    data_path : str
        The directory containing the data-files for the relevant contours.
    raw_contours : dict
        A persistent dictionary of contours; the keys are the contour names,
        and the values are the clicked points in the flatmaps.
    """
    import pyrsistent as pyr
    contours = load_contours(rater, sid, chirality,
                             save_path=save_path,
                             vc_contours=vc_contours)
    data_path = to_data_path(rater, sid, save_path=save_path)
    return (data_path, pyr.pmap(contours))
@pimms.calc('cortex')
def load_cortex(sid, hemisphere, chirality):
    """Loads the requested hemisphere object from the requested HCP subject.
    
    Parameters
    ----------
    sid : int
        The subject ID of the HCP-subject whose data is to be loaded.
    hemisphere : str
        The name of the HCP subject's hemisphere that is to be loaded.
    chirality : str
        Either `'lh'` if `hemisphere` is a left hemisphere or `'rh'` if it is
        a right hemisphere. This is calculated by `parse_chirality`.
        
    Outputs
    -------
    cortex : neuropythy Cortex
        The neuropythy cortex object, representing the subject's hemisphere.
    """
    import neuropythy as ny
    data = ny.data['hcp_lines']
    # First, check if the subject is an excluded subject or not.
    if (sid,chirality, 'mean') in data.exclusions:
        raise RuntimeError(f"excluded hemisphere: {sid}/{chirality}")
    # Otherwise grab them.
    sub = data.subjects[sid]
    hem = sub.hemis[hemisphere]
    return (hem,)
@pimms.calc('flatmap')
def calc_flatmap(cortex):
    """Calculates the flatmap used for annotation.
    
    Parameters
    ----------
    cortex : neuropythy Cortex
        The cortex object whose annotations are being examined.
    
    Outputs
    -------
    flatmap : neuropythy Mesh
        A 2D mesh of the flattened cortex.
    """
    return (op_flatmap(cortex),)
@pimms.calc('v3v_contour')
def load_v3v_contour(sid, chirality):
    """Loads the V3-ventral contour from the HCP-lines dataset.
    
    Parameters
    ----------
    sid : int
        The HCP subject-ID of the subject to load.
    chirality : 'lh' or 'rh'
        The hemisphere (`'lh'` or `'rh'`) to load.
        
    Outputs
    -------
    v3v_contour : NumPy array
        A NumPy array of the points in the V3-ventral contour.
    """
    sdat = subject_data[(sid, chirality)]
    v3v = sdat['v123']['V3_ventral']
    v3v = np.array(v3v)
    # This is the v3-ventral contour.
    return (v3v,)
def _extend_contour(pts, d=100):
    (u, v) = (pts[:,0] - pts[:,1], pts[:,-1] - pts[:,-2])
    u /= np.sqrt(np.sum(u**2))
    v /= np.sqrt(np.sum(v**2))
    pts_ext = [(u*d + pts[:,0])[:,None], pts, (v*d + pts[:,-1])[:,None]]
    pts_ext = np.hstack(pts_ext)
    return pts_ext
@pimms.calc('preproc_contours', 'ext_contours', 'outer_sources')
def calc_extended_contours(rater, raw_contours, chirality, v3v_contour,
                           vc_contours, meanrater=meanrater):
    """Creates and extended hV4/VO1 boundary contour.
    
    The contours are extended by adding points to either end that are a 
    distance of 100 map units from the respective endpoints at an angle of 180
    degrees to the immediate-next interior point.
    
    Parameters
    ----------
    raw_contours : dict-like
        The dictionary of contours.
    v3v_contour : NumPy array
        The `(2 x N)` matrix of points in the V3-ventral contour.
    meanrater : str
        The name of the mean rater (default is `'mean'`).
        
    Outputs
    -------
    preproc_contour : dict
        A dictionary whose keys are the same as in `raw_contours` but that have
        been mildly preprocessed: all contours are ordered starting from the
        end nearest the V3-ventral boundary. Additionally the `'outer'` contour
        has been added, consisting of the hV4-outer, V3-ventral, and VO-outer
        contours conjoined, starting at the hV4-outer and ending with the
        VO-outer.
    ext_contours : dict
        A dictionary whose keys are the same as in `preproc_contour` but whose
        ends have been extended by 100 map units in the same directoin as the
        ending segments.
    outer_sources : array of str
        A list whose elements give the origin of each point in the `'outer'`
        contour. Each element will be either `'hV4_outer'`, `'VO_outer'`, or
        `'V3v'`.
    """
    contours = dict(raw_contours)
    if any(k not in contours for k in ('outer', 'hV4_VO1', 'VO1_VO2')):
        contours['V3_ventral'] = v3v_contour
        # Both the hV4-VO1 contour need to be re-ordered to be starting near
        # the V3-ventral contour and ending far from it.
        v3v_end = v3v_contour[:,0]
        for c in ('hV4_VO1', 'VO1_VO2'):
            pts = contours[c]
            ends = pts[:,[0,-1]].T
            dists = [np.sqrt(np.sum((end - v3v_end)**2)) for end in ends]
            if dists[1] < dists[0]:
                pts = np.fliplr(pts)
                contours[c] = pts
        # Make the outer contour:
        outer = [np.fliplr(contours['hV4_outer']),
                 np.fliplr(v3v_contour),
                 contours['VO_outer']]
        outer_sources = (['hV4_outer'] * (outer[0].shape[1] + 1) + 
                         ['V3v'] * outer[1].shape[1] +
                         ['VO_outer'] * (outer[2].shape[1] + 1))
        outer = np.hstack(outer)
        outer_sources = np.array(outer_sources)
        contours['outer'] = outer
    else:
        outer_sources = None
    # Now make the extended contours:
    ext_contours = {k:_extend_contour(c) for (k,c) in contours.items()}
    # And return!
    return (contours, ext_contours, outer_sources)
def _cross_isect(segs1, segs2, rtol=1e-05, atol=1e-08):
    from neuropythy.geometry.util import segment_intersection_2D
    (segs1, segs2) = (np.asarray(segs1), np.asarray(segs2))
    (n1, n2) = (segs1.shape[1] - 1, segs2.shape[1] - 1)
    ii1 = np.concatenate([[k]*n2 for k in range(n1)])
    ii2 = np.arange(n2)
    ii2 = np.concatenate([ii2]*n1)
    pts = segment_intersection_2D([segs1[...,ii1], segs1[...,ii1+1]],
                                  [segs2[...,ii2], segs2[...,ii2+1]],
                                  inclusive=True)
    pts = np.asarray(pts)
    ii = np.all(np.isfinite(pts), axis=0)
    (ii1, ii2, pts) = (ii1[ii], ii2[ii], pts[:,ii])
    # We have to be careful with sorting: if two of the ii1 segments are
    # the same, we want to order them by the distance from the segment
    # start.
    seglen2s = np.sum((segs1[:, :-1] - segs1[:, 1:])**2, axis=0)
    ds = [np.sum(seglen2s[:ii1[ii]]) + np.sum((segs1[:, ii1[ii]] - pts[:,ii])**2)
          for ii in range(len(ii1))]
    ii = np.argsort(ds)
    (ii1, ii2, pts) = (ii1[ii], ii2[ii], pts[:,ii])
    # See if we have accidentally created identical points.
    for ii in reversed(range(pts.shape[1] - 1)):
        if np.isclose(pts[:,ii], pts[:,ii+1], atol=atol, rtol=rtol).all():
            ii1 = np.delete(ii1, ii+1)
            ii2 = np.delete(ii2, ii+1)
            pts = np.delete(pts, ii+1, axis=-1)
    return (ii1, ii2, pts)
def _closer(x, a, b):
    da2 = np.sum((x - a)**2)
    db2 = np.sum((x - b)**2)
    return (da2 < db2)
@pimms.calc('contours', 'boundaries')
def calc_normalized_contours(sid, hemisphere, rater, outer_sources,
                             preproc_contours, ext_contours):
    """Normalizes the raw contours and converts them into path-traces.
    """
    hv4_vo1 = ext_contours['hV4_VO1']
    vo1_vo2 = ext_contours['VO1_VO2']
    outer = ext_contours['outer']
    # We want to find the intersections between hV4-VO1 and the outer boundary.
    # There must be 3; any other number is an error.
    (hii, oii, pts_ho) = _cross_isect(hv4_vo1, outer)
    if len(hii) > 3:
        # We need to figure out which three intersections of the hV4-VO1 border
        # and the outer border we're going to use.
        # One of these intersections must be near the V3v border, so we use the
        # outer sources to figure that out.
        if outer_sources is None:
            raise RuntimeError(f"{len(hii)} hV4-VO1 / Outer intersections for "
                               f"{rater}/{sid}/{hemisphere}, which has no "
                               f"outer_sources data")
        v3v_start = np.where(outer_sources == 'V3v')[0][0]
        voo_start = np.where(outer_sources == 'VO_outer')[0][0]
        v3v_dist = np.abs(voo_start - oii)
        v3v_dist[(v3v_dist >= v3v_start) & (v3v_dist < voo_start)] = 0
        isect_ii = np.argmin(v3v_dist)
        # Now we just want the closest value in each direction from that point.
        v3v_dist = oii[isect_ii] - oii
        ii = v3v_dist > 0
        if np.sum(ii) == 0:
            raise ValueError(f"no v3v>0 intersections found for "
                             f"{rater}/{sid}/{hemisphere}")
        isect_hv4 = np.where(ii)[0][np.argmin(v3v_dist[ii])]
        ii = v3v_dist < 0
        if np.sum(ii) == 0:
            raise ValueError(f"no v3v<0 intersections found for "
                             f"{rater}/{sid}/{hemisphere}")
        isect_vo = np.where(ii)[0][np.argmax(v3v_dist[ii])]
        # Put these together into the (hii, oii, pts_ho) variables.
        hii = hii[[isect_ii, isect_hv4, isect_vo]]
        oii = oii[[isect_ii, isect_hv4, isect_vo]]
        pts_ho = pts_ho[:,[isect_ii, isect_hv4, isect_vo]]
        ii = np.argsort(hii)
        (hii,oii,pts_ho) = (hii[ii], oii[ii], pts_ho[:,ii])
    elif len(hii) != 3:
        raise RuntimeError(f"{len(hii)} hV4-VO1 / Outer intersections for "
                           f"{rater}/{sid}/{hemisphere}")
    # These intersections will have been sorted by distance along the hV4-VO1
    # boundary, so the first one must necessarily be the "V3-ventral" end.
    hii_v3v = hii[0]
    pii = [1,2] if oii[1] < oii[2] else [2,1]
    (oii_hv4, oii_vo) = oii[pii]
    (hii_hv4, hii_vo) = hii[pii]
    # Shorten the hV4-extended line to just the relevant part.
    if   hii_hv4 < hii_vo: (hii_end, pii_end) = (hii_vo,  pii[1])
    elif hii_hv4 > hii_vo: (hii_end, pii_end) = (hii_hv4, pii[0])
    else:
        x_hv4 = pts_ho[:,pii[0]]
        x_vo  = pts_ho[:,pii[1]]
        x_hii = hv4_vo1[:,hii_hv4]
        if _closer(x_hii, x_hv4, x_vo): (hii_end, pii_end) = (hii_vo,  pii[1])
        else:                           (hii_end, pii_end) = (hii_hv4, pii[0])
    hv4_vo1_norm = np.hstack([pts_ho[:,[0]],
                              hv4_vo1[:, (hii_v3v + 1):hii_end],
                              pts_ho[:,[pii_end]]])
    outer_norm = np.hstack([pts_ho[:, [pii[0]]],
                            outer[:, (oii[pii[0]] + 1):oii[pii[1]]],
                            pts_ho[:, [pii[1]]]])
    # Now, we can find the intersection of outer and VO1/VO2.
    (vii, uii, pts_vu) = _cross_isect(vo1_vo2, outer_norm)
    if len(vii) != 2:
        raise RuntimeError(f"{len(vii)} VO1-VO2 / Outer intersections for "
                           f"{rater}/{sid}/{hemisphere}")
    vo1_vo2_norm = np.hstack([pts_vu[:,[0]], 
                              vo1_vo2[:, (vii[0]+1):(vii[1] + 1)],
                              pts_vu[:,[1]]])
    contours = {'VO1_VO2': vo1_vo2_norm,
                'hV4_VO1': hv4_vo1_norm,
                'outer':   outer_norm}
    # Now make the boundaries.
    hv4_b = np.hstack([
        # First, hV4 from V3-ventral to the hV4-outer.
        pts_ho[:, [0]],
        hv4_vo1[:, (hii_v3v + 1):hii_hv4],
        pts_ho[:, [pii[0]]],
        # Next, outer from this point to back to the hV4-V3-ventral point.
        outer[:, (oii[pii[0]] + 1):oii[0]]])
    vo1_b = np.hstack([
        # First, VO1-VO2.
        vo1_vo2_norm,
        # Then the outer from there to the hV4-VO1 contour.
        outer_norm[:, (uii[1] + 1):],
        # Then the hV4-VO1 boundary.
        pts_ho[:, [pii[1]]],
        np.fliplr(hv4_vo1[:, (hii_v3v + 1):hii_vo]),
        pts_ho[:, [0]],
        # Finally, the outer boundary from hV4-VO1 to VO1-VO2.
        outer[:, (oii[0] + 1):(uii[0] + oii[pii[0]] + 1)]])
    vo2_b = np.hstack([
        # First the outer around VO2.
        outer_norm[:, (uii[0] + 1):(uii[1] + 1)],
        # Then, VO1-VO2.
        np.fliplr(vo1_vo2_norm)])
    boundaries = {'hV4': hv4_b, 'VO1': vo1_b, 'VO2': vo2_b}
    return (contours, boundaries)
@pimms.calc('traces')
def calc_traces(flatmap, boundaries, contours):
    """Calculates path-traces from the flatmap and boundaries.
    """
    mp = flatmap.meta_data['projection']
    traces = {k: ny.path_trace(mp, pts, closed=True)
              for (k,pts) in boundaries.items()}
    for (k,pts) in contours.items():
        traces[k] = ny.path_trace(mp, pts, closed=False)
    return (traces,)
vc_plan = pimms.plan(
    parse_chirality=parse_chirality,
    load_contours=calc_load_contours,
    load_v3v_contour=load_v3v_contour,
    extend_contours=calc_extended_contours,
    normalize_contours=calc_normalized_contours,
    load_cortex=load_cortex,
    flatmap=calc_flatmap,
    traces=calc_traces)

# Visualization ################################################################
raw_colors = {
    'hV4_outer': (0.5, 0,   0),
    'hV4_VO1':   (0,   0.3, 0),
    'VO_outer':  (0,   0.4, 0.6),
    'VO1_VO2':   (0,   0,   0.5)}
preproc_colors = {
    'hV4_outer': (0.7, 0,   0),
    'hV4_VO1':   (0,   0.5, 0),
    'VO_outer':  (0,   0.6, 0.8),
    'VO1_VO2':   (0,   0,   0.7),
    'V3_ventral':(0.7, 0,   0.7),
    'outer':     (0.7, 0.7, 0, 0)}
ext_colors = {
    'hV4_outer': (1,   0,   0),
    'hV4_VO1':   (0,   0.8, 0),
    'VO_outer':  (0,   0.9, 1),
    'VO1_VO2':   (0,   0,   1),
    'V3_ventral':(0.8, 0,   0.8),
    'outer':     (0.8, 0.8, 0.8, 1)}
boundary_colors = {
    'hV4': (1, 0.5, 0.5),
    'VO1': (0.5, 1, 0.5),
    'VO2': (0.5, 0.5, 1)}

def plot_vc_contours(dat, raw=None, ext=None, preproc=None,
                     contours=None, boundaries=None,
                     figsize=(2,2), dpi=(72*5), axes=None, 
                     flatmap=True, lw=1, color='prf_polar_angle',
                     mask=('prf_variance_explained', 0.05, 1)):
    """Plots a rater's ventral ccontours on the cortical flatmap.

    `plot_vc_contours(data)` plots a flatmap of the visual cortex for the
    subject whose data is contained in the parameter `data`. This parameter must
    be an output dictionary of the `vc_plan` plan. Contours can be drawn on the
    flatmap by providing one or more of the optional arguments `raw`, `ext`,
    `preproc`, `contours`, and `boundaries`. If any of these is set to `True`,
    then that set of contours is drawn on the flatmap with a default
    color-scheme. Alternately, if a dictionary is given, its keys must be the
    contour names and its values must be colors.

    Parameters
    ----------
    data : dict
        An output dictionary from the `vc_plan` plan.
    raw : boolean or dict, optional
        Whether and how to plot the raw contours (i.e., the contours as drawn by
        the raters).
    ext : boolean or dict, optional
        Whether and how to plot the extended raw contours.
    preproc : boolean or dict, optional
        Whether and how to plot the preprocessed contours.
    contours : boolean or dict, optional
        Whether and how to plot the processed contours.
    boundaries : boolean or dict, optional
        Whether and how to plot the final boundaries.
    figsize : tuple of 2 ints, optional
        The size of the figure to create, assuming no `axes` are given. The
        default is `(2,2)`.
    dpi : int, optional
        The number of dots per inch to given the created figure. If `axes` are
        given, then this is ignored. The default is 360.
    axes : matplotlib axes, optional
        The matplotlib axes on which to plot the flatmap and contours. If this
        is `None` (the default), then a figure is created using `figsize` and
        `dpi`.
    flatmap : boolean, optional
        Whether or not to draw the flatmap. The default is `True`.
    lw : int, optional
        The line-width to use when drawing the contours. The default is 1.
    color : str or flatmap property, optional
        The color to use in the flatmap plot; this option is passed directly to
        the `ny.cortex_plot` function. The default is `'prf_polar_angle'`.
    mask : mask-like, optional
        The mask to use when plotting the color on the flatmap. This option is
        passed directly to the `ny.cortex_plot` function. The default value is
        `('prf_variance_explained', 0.05, 1)`.

    Returns
    -------
    matplotlib.Figure
        The figure on which the plot was made.
    """
    # Make the figure.
    if axes is None:
        (fig,ax) = plt.subplots(1,1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(0,0,1,1,0,0)
    else:
        ax = axes
        fig = ax.get_figure()
    # Plot the flatmap.
    if flatmap:
        fmap = dat['flatmap']
        ny.cortex_plot(fmap, color=color, mask=mask, axes=ax)
    # Plot the requested lines:
    if raw is not None:
        if raw is True: raw = raw_colors
        for (k,v) in dat['raw_contours'].items():
            c = raw.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if preproc is not None:
        if preproc is True: preproc = preproc_colors
        for (k,v) in dat['preproc_contours'].items():
            c = preproc.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if ext is not None:
        if ext is True: ext = ext_colors
        for (k,v) in dat['ext_contours'].items():
            c = ext.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if contours is not None:
        if contours is True: contours = ext_colors
        for (k,v) in dat['contours'].items():
            c = contours.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if boundaries is not None:
        if boundaries is True: boundaries = boundary_colors
        for (k,v) in dat['boundaries'].items():
            c = boundaries.get(k, 'w')
            x = np.concatenate([v[0], [v[0][0]]])
            y = np.concatenate([v[1], [v[1][0]]])
            ax.plot(x, y, '-', color=c, lw=lw)
    ax.axis('off')
    return fig
