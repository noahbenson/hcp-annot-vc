################################################################################
# io.py
#
# Input and output tools for the HCP visual cortex contours.
# by Noah C. Benson <nben@uw.edu>

import sys, os, pimms, json
from collections.abc import Mapping
import numpy as np
import pyrsistent as pyr
import neuropythy as ny

from .analysis import (vc_plan, vc_contours, vc_contours_meanrater, meanrater,
                       all_traces, to_data_path, save_contours, load_contours)

def guess_raters(path):
    """Returns a list of possible rater names in the given path.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    return [
        flnm for flnm in os.listdir(path)
        if not flnm.startswith('.')
        if os.path.isdir(os.path.join(path, flnm))]
def save_traces(traces, h, data_path, overwrite=True):
    """Saves a dictionary of traces to a particular directory.
    
    `save_traces(traces, h, data_path)` saves a series of files named
    `f"{h}.{key}_trace.json.gz"`, one per key-value pair in the `traces` dict,
    out to the `save_path`, which must be an existing directory.
    
    The optional parameter `overwrite` (default: `True`) can be set to `False`
    to require that the files are not overwritten if they already exist.
    """
    fls = {}
    for (k,tr) in traces.items():
        flnm = os.path.join(data_path, f'{h}.{k}_trace.json.gz')
        if overwrite or not os.path.isfile(flnm):
            # We need to clear the mesh field for this to work properly.
            tr = tr.copy(map_projection=tr.map_projection.copy(mesh=None))
            flnm = ny.save(flnm, ny.util.normalize(tr.normalize()), 'json')
        fls[k] = flnm
    return fls
def export_traces(rater, sid, h,
                  save_path='.',
                  load_path=None,
                  overwrite=True,
                  mkdir=True,
                  mkdir_mode=0o775,
                  vc_plan=vc_plan, 
                  vc_contours=vc_contours):
    """Calculates and saves the traces for a rater, subject, and hemisphere.

    This function is intended to be called with `tupcall` and `mprun` functions
    in order to process and save the traces of a single rater, subject, and
    hemisphere in the HCP annotation project. The first three arguments of the
    function are the `rater`, the `sid` (subject ID), and `h` (hemisphere).
    
    Parameters
    ----------
    rater : str
        The rater whose contours should be processed.
    sid : int
        The HCP subject ID of the subject whose contours should be processed.
    h : 'lh' or 'rh'
        The hemisphere that should be processed.
    save_path : directory name
        The directory to which this set of traces should be saved. Traces
        themselves are saved into a directory equivalen to
        `os.path.join(save_path, rater, str(sid))`.
    load_path : directory name, optional
        The directory from which traces should be loaded; if not provided, then
        defaults to the `save_path`.
    vc_plan : pimms calculation plan, optional
        The plan that is to be executed on the contours. This plan must produce
        an output value called `'traces'` that contains the traces to be saved
        to disk. The default is `hcpannot.vc_plan`.
    overwrite : boolean, optional
        Whether to overwrite the files, should they exist. The default is
        `True`.
    vc_contours : dict
        A dictionary whose keys are the names of the contours and whose values
        are format strings for the filenames that store the given contours. The
        contours correspond to those drawn by raters using the annotation tool.
        The default is `hcpannot.analysis.vc_contours`.
        
    Returns
    -------
    dict
        A dictionary whose keys are the contour names and whose values are the
        filenames to which the associated trace was saved.
    """
    if load_path is None:
        load_path = save_path
    dat = vc_plan(rater=rater, sid=sid, hemisphere=h,
                  save_path=load_path,
                  vc_contours=vc_contours)
    h = dat['chirality']
    data_path = to_data_path(rater, sid, save_path=save_path)
    if not os.path.isdir(data_path) and mkdir:
        os.makedirs(data_path, mode=mkdir_mode)
    return save_traces(dat['traces'], h, data_path,
                       overwrite=overwrite)
def load_traces(rater, sid, h,
                save_path='.',
                traces=all_traces):
    """Loads and returns a dict of traces, as saved by `export_traces`.

    `load_traces(rater, sid, h, save_path)` returns a dictionary whose keys are
    the names of traces and whose values are the traces that were saved out by
    the `export_traces(rater, sid, h, save_path)` function.
    """
    data_path = to_data_path(rater, sid, save_path=save_path)
    h = ny.to_hemi_str(h.split('_')[0])
    r = {}
    for flnm in os.listdir(data_path):
        if not flnm.endswith('_trace.json.gz'): continue
        (hh, k) = flnm.split('.')[:2]
        k = k.split('_')[:-1]
        k = '_'.join(k)
        if hh != h or k not in traces: continue
        tr = ny.load(os.path.join(data_path, flnm))
        # This is a bug? #TODO
        if isinstance(tr, dict):
            pts = tr.pop('points')
            if h == 'lh': pts = np.fliplr(pts)
            mpj = tr.pop('map_projection')
            tr = ny.geometry.PathTrace(mpj, pts, **tr)
        r[k] = tr
    return r
def export_paths(rater, sid, h,
                 save_path='.',
                 load_path=None,
                 overwrite=True,
                 mkdir=True,
                 mkdir_mode=0o775):
    """Calculates and saves the paths for a rater, subject, and hemisphere.

    This function is intended to be called with `tupcall` and `mprun` functions
    in order to process and save the paths of a single rater, subject, and
    hemisphere in the HCP annotation project. The first three arguments of the
    function are the `rater`, the `sid` (subject ID), and `h` (hemisphere). The
    traces must already be calculated and saved to the given `save_path` prior
    to calling this function.
    
    Parameters
    ----------
    rater : str
        The rater whose traces should be processed.
    sid : int
        The HCP subject ID of the subject whose traces should be processed.
    h : 'lh' or 'rh'
        The hemisphere that should be processed.
    save_path : directory name, optional
        The directory to which this set of traces should be saved. Traces
        themselves are saved into a directory equivalen to
        `os.path.join(save_path, rater, str(sid))`. The default is `'.'`.
    load_path : directory name, optional
        The directory from which traces should be loaded; if not provided, then
        defaults to the `save_path`.
    overwrite : boolean, optional
        Whether to overwrite the files, should they exist. The default is
        `True`.
        
    Returns
    -------
    dict
        A dictionary whose keys are the contour names and whose values are the
        filenames to which the associated path was saved.
    """
    if load_path is None:
        load_path = save_path
    trs = load_traces(rater, sid, h, save_path=load_path)
    data_path = to_data_path(rater, sid, save_path=save_path)
    sub = ny.data['hcp_lines'].subjects[sid]
    hem = sub.hemis[h]
    r = []
    for (k,tr) in trs.items():
        flnm = os.path.join(data_path, f'{h}.{k}_path.json.gz')
        if not overwrite and os.path.isfile(flnm): continue
        if not os.path.isdir(data_path) and mkdir:
            os.makedirs(data_path, mode=mkdir_mode)
        p = tr.to_path(hem)
        ny.save(flnm, p.addresses)
        r.append(flnm)
    return r
def load_paths(rater, sid, h, save_path='.',
               paths=('hV4', 'VO1', 'VO2')):
    """Loads and returns a dict of paths, as saved by `export_paths`.

    `load_paths(rater, sid, h, save_path)` returns a dictionary whose keys are
    the names of paths and whose values are the paths that were saved out by the
    `export_paths(rater, sid, h, save_path)` function.
    """
    data_path = to_data_path(rater, sid, save_path=save_path)
    sub = ny.data['hcp_lines'].subjects[sid]
    hem = sub.hemis[h]
    h = ny.to_hemi_str(h.split('_')[0])
    r = {}
    for flnm in os.listdir(data_path):
        if not flnm.endswith('_path.json.gz'): continue
        (hh, k) = flnm.split('.')[:2]
        if hh != h: continue
        k = '_'.join(k.split('_')[:-1])
        addr = ny.load(os.path.join(data_path, flnm))
        if k in paths:
            # We need to make sure the addresses are closed.
            faces = np.asarray(addr['faces'])
            barys = np.asarray(addr['coordinates'])
            (f0,f1) = (faces[:,0], faces[:,-1])
            (b0,b1) = (barys[:,0], barys[:,-1])
            if not ((f0 == f1).all() and np.isclose(b0, b1).all()):
                faces = np.hstack([faces, f0[:,None]])
                barys = np.hstack([barys, b0[:,None]])
                addr = {'faces': faces, 'coordinates': barys}
        p = ny.geometry.Path(hem, addr)
        r[k] = p
    return r
def export_means(sid, h,
                 save_path='.',
                 load_path=None,
                 raters=None,
                 npoints=500,
                 overwrite=True,
                 mkdir=True,
                 mkdir_mode=0o775,
                 vc_contours=vc_contours_meanrater,
                 meanrater=meanrater):
    """Calculates and saves the mean contours for a subject and hemisphere.

    This function is intended to be called with `tupcall` and `mprun` functions
    in order to process and save the traces of a single subject and hemisphere
    in the HCP annotation project. The first two arguments of the function are
    the `sid` (subject ID) and `h` (hemisphere).
    
    Parameters
    ----------
    sid : int
        The HCP subject ID of the subject whose contours should be processed.
    h : 'lh' or 'rh'
        The hemisphere that should be processed.
    save_path : directory name, optional
        The directory to which this set of traces should be saved. Traces
        themselves are saved into a directory equivalen to
        `os.path.join(save_path, rater, str(sid))`. The default is `'.'`.
    load_path : directory name, optional
        The directory from which traces should be loaded; if not provided, then
        defaults to the `save_path`.
    raters : None or list of str, optional
        Either a list of raters that are to be included in the mean contours
        or `None` if all available raters should be included. The default is
        `None`.
    npoints : int, optional
        The number of points to divide each contour up into when averaging
        across raters. The default is 500.
    overwrite : boolean, optional
        Whether to overwrite the files, should they exist. The default is
        `True`.
    vc_contours : dict, optional
        A dictionary whose keys are the names of the contours and whose values
        are format strings for the filenames that store the given contours. The
        contours correspond to those drawn by raters using the annotation tool.
        The default is `hcpannot.analysis.vc_contours_meanrater`.
    meanrater : str, optional
        The name to use for the mean rater. By default this is the value in
        `hcpannot.analysis.meanrater`, which is `'mean'`.

    Returns
    -------
    dict
        A dictionary whose keys are the contour names and whose values are a
        two-tuple of `(filename, raterlist)` where `filename` is the path to
        which the associated mean contour was saved and `raterlist` is a list
        of the raters whose contours were averaged in order to make the contour
        that was exported.
    """
    if load_path is None:
        load_path = save_path
    # This is where we will load and/or eventually save these contour files.
    data_path = to_data_path(meanrater, sid, save_path=save_path)
    # First, check if these data already exist (if we're not overwriting).
    if not overwrite and os.path.isdir(data_path):
        try:
            # We use save_path here because we are checking for complete results
            # that have already been calculated and saved.
            cs = load_contours(meanrater, sid, h, save_path,
                               vc_contours=vc_contours)
            if len(cs) > len(vc_contours):
                cs = {c:v for (c,v) in cs if c in vc_contours}
            if all(c in cs for c in vc_contours):
                return cs
        except Exception:
            pass
    # First things first: we need to load in the traces of all raters.
    if raters is None:
        raters = guess_raters(load_path)
    trs = {}
    for rater in raters:
        # Try loading the traces.
        try:
            tr = load_traces(rater, sid, h, load_path)
        except Exception:
            tr = ()
        if len(tr) == 0: continue
        # Turn these into linspaced points.
        trs[rater] = {k: v.copy(points=v.curve.linspace(npoints))
                      for (k,v) in tr.items()}
    # Process these into a mean map of contours.
    meantrs = {}
    rcounts = {}
    for k in vc_contours:
        trlist = [u for u in trs.values() if k in u]
        if len(trlist) == 0:
            raise ValueError(f"no raters found for contour {k}")
        meantrs[k] = np.mean([u[k].points for u in trlist], axis=0)
        rcounts[k] = len(trlist)
    # Save the means; this gives us back a dict of filenames.
    res = save_contours(meanrater, sid, h, meantrs, save_path,
                        overwrite=overwrite,
                        vc_contours=vc_contours,
                        mkdir=mkdir,
                        mkdir_mode=mkdir_mode)
    # Finally, process the results into a dict whose values are tuples
    # of the (filename, ratercount).
    res = {k: (v,rcounts[k]) for (k,v) in res.items()}
    return res
def calc_surface_areas(rater, sid, h,
                       load_path='.',
                       boundaries=('hV4', 'VO1', 'VO2')):
    """Returns the surface area of each visual area as a dict.

    `calc_surface_areas(rater, sid, h, save_path)` returns a dict whose keys are
    the names of the boundary paths for the given rater, subject ID, and
    hemisphere, and whose values are the surface areas of each boundary.
    
    The optional argument `boundaries` may be passed to specify that the surface
    areas of specific boundaries be computed; the default is
    `('hV4', 'VO1', 'VO2')`.
    """
    ps = load_paths(rater, sid, h, load_path)
    hem = None
    for k in boundaries:
        if k not in ps:
            raise ValueError(f"path {k} not found for {rater}/{sid}/{h}")
        elif hem is None:
            hem = ps[k].surface
    # This is the surface area that is calculated by the surface_area map;
    # i.e., paths don't know about the vertex map that uses NaNs along the
    # medial surface, it just knows about triangle areas.
    c = np.nansum(hem.midgray_surface.face_areas)
    r = {k: (np.nan if p.surface_area is None else p.surface_area['midgray'])
         for (k,p) in ps.items()
         if k in ('hV4', 'VO1', 'VO2')}
    # Make sure we calculated the correct internal (not external) area.
    r = {k: (v if v < (c - v) else (c - v)) for (k,v) in r.items()}
    # This value is the cortical surface area excluding the medial wall.
    r['cortex'] = np.nansum(hem.prop('midgray_surface_area'))
    # Add in the rest of the ID stuff and return.
    r['rater'] = rater
    r['sid'] = sid
    r['hemisphere'] = h
    return r
def export_labels(raters, sid,
                  save_path='.',
                  load_path=None,
                  paths=('hV4', 'VO1', 'VO2'),
                  overwrite=True,
                  mkdir=True,
                  mkdir_mode=0o775,
                  output_weights=False,
                  output_volume=False,
                  exit_on_finish=False):
    """Exports the labels for the visual areas hV4, VO1, and VO2.

    Exports the requested labels to disk. The labels are obtained by loading the
    paths via `load_paths` then converting them to labels.

    Parameters
    ----------
    raters : str or list of str
        The rater or raters to export the labels for.
    sid : int
        The HCP subject ID of the subject whose contours should be processed.
    save_path : directory name, optional
        The directory to which this set of labels should be saved. Labels
        themselves are saved into a directory equivalen to
        `os.path.join(save_path, rater)`. The default is `'.'`.
    load_path : directory name, optional
        The directory from which traces should be loaded; if not provided, then
        defaults to the `save_path`.
    paths : str or list of str
        Either a list of path names that are to be included in the exported
        files or a single path name. If a dictionary is given, then the paths
        are assumed to be already loaded.
    """
    # If load path isn't provided, we guess it.
    if load_path is None:
        load_path = save_path
    # Iterate through the raters.
    if isinstance(raters, str):
        raters = [raters]
    if isinstance(paths, str):
        paths = (paths,)
    for rater in raters:
        try:
            # We want to start by generating and saving the labels for the
            # cortical surface.
            props = []
            for h in ['lh', 'rh']:
                # Make sure we need to do the work!
                path = os.path.join(save_path, rater)
                filename = os.path.join(path, f'{h}_{sid}.mgz')
                if not overwrite and os.path.isfile(filename):
                    props.append(filename)
                    continue
                if isinstance(paths, Mapping):
                    ps = paths[h]
                else:
                    ps = load_paths(rater, sid, h, load_path, paths=paths)
                lbls = []
                for k in paths:
                    p = ps[k]
                    lbl = p.label
                    if np.sum(lbl) > np.sum(1 - lbl):
                        lbl = 1 - lbl
                    lbls.append(lbl)
                lbls = np.array(lbls)
                # Add the zero label, which is the probability of not being in a
                # label.
                nolbl = np.min(
                    [np.zeros(lbls.shape[1]), 1 - np.sum(lbls, axis=0)],
                    axis=0)
                lbls = np.concatenate([nolbl[None,:], lbls])
                # Make the directory for outputs if need-be.
                if mkdir and not os.path.exists(path):
                    os.mkdirs(path, mkdir_mode)
                if output_weights:
                    overlap_flnm = os.path.join(path, f'{h}_{sid}_weights.mgz')
                    ny.save(overlap_flnm, lbls)
                # We don't want to label anything as part of an area if the
                # label value is less than or equal to 0.5.
                lbls[lbls <= 0.5] = 0
                # Before we find the argmax, we want to include a row 0 such
                # that any vertex not in a visual area will be given label 0.
                lbls[0,:] = 0.25
                # Now find the argmax; 0 indicates none of the labels.
                lbl = np.argmax(lbls, axis=0).astype(int)
                # Save this file out!
                ny.save(filename, lbl)
                # Save this for volume interpolation also.
                props.append(lbl)
            # Now, we also want to interpolate to the volume and save that out.
            if output_volume:
                filename = os.path.join(path, f'{sid}.mgz')
                if overwrite or not os.path.isfile(filename):
                    props = tuple(
                        ny.load(flnm) if isinstance(flnm, str) else flnm
                        for flnm in props)
                    sub = ny.hcp_subject(sid)
                    template_im = ny.image_clear(sub.images['ribbon'])
                    im = sub.cortex_to_image(
                        props, template_im,
                        method='nearest')
                    ny.save(filename, im)
        except Exception as e:
            print(f"  - Failure for rater {rater}: {e}")
    # Exit!
    if exit_on_finish:
        #print(f"Exiting: {os.getpid()}: {rater} / {sid}", file=sys.stderr)
        sys.exit(0)

