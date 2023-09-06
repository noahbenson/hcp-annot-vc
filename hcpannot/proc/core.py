################################################################################
# proc/core.py
#
# The core and shared plan components for the HCP annotation project.


# Dependencies #################################################################

import os
from pathlib import Path

import numpy as np
import neuropythy as ny
import pimms

from ..config import (
    procdata,
    to_data_path,
    cortex,
    labelkey,
    meanrater)
from ..core import (
    op_flatmap)
from ..io import (
    load_contours, load_traces, load_paths, load_labels, load_reports,
    save_contours, save_traces, save_paths, save_labels, save_reports)


# The Initialization Plan ######################################################
# This section defines the initialization plan, which includes all aspects of
# processing that is shared between regions.

# The calculations -------------------------------------------------------------
@pimms.calc('io_options')
def calc_io_options(overwrite=None, mkdir=True, mkdir_mode=0o775):
    """Gathers the save/load options for the plan into an options dictionary.

    Parameters
    ----------
    overwrite : boolean or None, optional
        Whether to overwrite files when processing data. If `True`, then all
        processing is repeated and new files are exported. If `False`, then no
        new calculations are performed and an exception is raised if any of the
        save files are not found. If `None` (the default) then data is loaded
        from disk when possible and recalculated when not.
    mkdir : boolean, optional
        Whether to make directories that do not exist when saving data. The
        default is `True`.
    mkdir_mode : int, optional
        The mode to given newly created directories if `mkdir` is True`. The
        default is `0o775`.
    """
    if overwrite is not None:
        overwrite = bool(overwrite)
    mkdir = bool(mkdir)
    mkdir_mode = int(mkdir_mode)
    return ({'overwrite': overwrite, 'mkdir': mkdir, 'mkdir_mode': mkdir_mode},)
@pimms.calc('chirality')
def calc_parse_chirality(hemisphere):
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
    return ny.to_hemi_str(hemisphere.split('_')[0])
@pimms.calc('contours')
def calc_load_contours(rater, sid, chirality, load_path, region):
    """Load the contours for a rater, subject, and hemisphere.
    
    Parameters
    ----------
    rater : str
        The rater whose contours are to be loaded.
    sid : int
        The HCP subject-ID of the subject to load.
    chirality : 'lh' or 'rh'
        The hemisphere (`'lh'` or `'rh'`) to load.
    load_path : str
        The path of the `save/` directory from the `hcp-annot-vc:data`
        repository.
    region : str
        The region name for the contours being loaded.

    Outputs
    -------
    contours : dict
        A persistent dictionary of contours; the keys are the contour names,
        and the values are the clicked points in the flatmaps.
    """
    contour_filenames = procdata(region, 'contours')
    contours = load_contours(
        rater, sid, chirality, contour_filenames,
        load_path=load_path)
    return (contours,)
@pimms.calc('cortex')
def calc_load_cortex(sid, hemisphere, chirality):
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
    # First, check if the subject is an excluded subject or not.
    data = ny.data['hcp_lines']
    if (sid, chirality, 'mean') in data.exclusions:
        raise RuntimeError(f"excluded hemisphere: {sid}/{chirality}")
    # Otherwise grab them.
    return (cortex(sid, hemisphere),)
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
@pimms.calc('contours')
def calc_mean_contours(sid, chirality, load_path, save_path, region, io_options,
                       npoints=500, min_raters=3, source_raters=None,
                       rater=meanrater):
    """Load or calculate the contours for the mean rater.
    
    Parameters
    ----------
    sid : int
        The HCP subject-ID of the subject to load.
    chirality : 'lh' or 'rh'
        The hemisphere (`'lh'` or `'rh'`) to load.
    load_path : str
        The path where the individual raters' traces have been stored. If the
        trace files are stored in a directory corresponding to
        `{rootpath}/traces/{rater}/{sid}/` then the provided `load_path` should
        me `{rootpath}/traces/`.
    save_path : directory name
        The directory to which this set of contours, traces, and other products
        should be saved. Traces themselves are saved into a directory equivalen
        to `os.path.join(save_path, rater, str(sid))`.
    region : str
        The region name for the contours being loaded.
    source_raters : sequence of str or None, optional
        The raters whose contours should be averaged. If `None` (the default),
        then the raters in the `procdata` are used.
    rater : str, optional
        The name for the mean rater whose contours are to be loaded or computed.
        By default this is `'mean'`.
    npoints : int, optional
        The number of points to split each rater's trace into when averaging. By
        default this is 500.

    Outputs
    -------
    contours : dict
        A persistent dictionary of contours; the keys are the contour names,
        and the values are the average contours across raters.
    """
    h = chirality
    overwrite = io_options.get('overwrite', None)
    if source_raters is None:
        source_raters = procdata(region, 'raters')
    # We first want to try to load them.
    contour_filenames = procdata(region, 'contours')
    means_path = os.path.join(save_path, 'means')
    traces_path = os.path.join(save_path, 'traces')
    contours = None
    if overwrite is not True:
        try:
            contours = load_contours(
                rater, sid, chirality, contour_filenames,
                load_path=means_path)
        except FileNotFoundError:
            pass
    if contours is None:
        # We need to load the traces, calculate the contours, then save them.
        if source_raters is None:
            from .config import raters_by_region
            source_raters = raters_by_region[region]
        source_traces = procdata(region, 'sources')
        traces = {}
        for rater in source_raters:
            try:
                traces[rater] = load_traces(
                    rater, sid, h, source_traces,
                    load_path=traces_path)
            except FileNotFoundError:
                pass
        if len(traces) < min_raters:
            raise RuntimeError(
                f"not enough raters ({len(traces)} found for contours")
        # We have loaded enough traces; now we average them into contours.
        contours = {}
        for name in source_traces.keys():
            trs = [
                trace_set[name].curve.linspace(npoints)
                for (rater,trace_set) in traces.items()]
            contours[name] = np.mean(trs, axis=0)
        if overwrite is not False:
            save_contours(
                meanrater, sid, h, contours,
                save_path=means_path,
                filenames=region,
                **io_options)
    return (contours,)


# The plan ---------------------------------------------------------------------
init_plan = pimms.plan(
    io_options      = calc_io_options,
    parse_chirality = calc_parse_chirality,
    load_contours   = calc_load_contours,
    load_cortex     = calc_load_cortex,
    flatmap         = calc_flatmap)
init_meanplan = pimms.plan(
    io_options      = calc_io_options,
    parse_chirality = calc_parse_chirality,
    calc_contours   = calc_mean_contours,
    load_cortex     = calc_load_cortex,
    flatmap         = calc_flatmap)


# The Data Plan ################################################################
# The data plan is a plan that requires another plandict as input and then
# finishes the path and label processing while handling the saving of data to
# disk and loading of data from disk when possible.

# Data from the trace plan -----------------------------------------------------
@pimms.calc('region', 'rater', 'sid', 'hemisphere', 'chirality', 'io_options')
def forward_constants(nested_data):
    """Retrieve various constants from the `nested_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    rater : str
        The rater whose contours are being processed.
    sid : int
        The HCP subject-ID of the subject being processed.
    hemisphere : str
        The hemisphere (usually `'lh'` or `'rh'`) being processed.
    chirality : 'lh' or 'rh'
        The hemisphere chirality (`'lh'` or `'rh'`).
    io_options : dict
        The input/output options such as whether to overwrite files.
    """
    return (
        nested_data['region'],
        nested_data['rater'],
        nested_data['sid'],
        nested_data['hemisphere'],
        nested_data['chirality'],
        nested_data['io_options'])
@pimms.calc('cortex')
def forward_cortex(nested_data):
    """Retrieve the `'cortex'` data from the `trace_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    cortex
        The neuropythy `Cortex` object on which contours are being processed.
    """
    return (nested_data['cortex'],)
@pimms.calc('flatmap')
def forward_flatmap(nested_data):
    """Retrieve the `'flatmap'` data from the `nested_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    flatmap
        The neuropythy 2D `Mesh` object on which contours are being processed.
    """
    return (nested_data['flatmap'],)
@pimms.calc('contours')
def forward_contours(nested_data):
    """Retrieve the `'contours'` data from the `nested_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    contours : dict
        The raw contours that are being processed.
    """
    return (nested_data['contours'],)
@pimms.calc('traces')
def forward_traces(nested_data):
    """Retrieve the `'traces'` data from the `nested_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    traces : dict
        The neuropythy path-trace objects that represent the processed / cleaned
        contours.
    """
    return (nested_data['traces'],)
@pimms.calc('paths')
def forward_paths(nested_data):
    """Retrieve the `'paths'` data from the `nested_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    paths : dict
        The neuropythy path objects that represent the processed / cleaned
        paths along the coritcal meshes.
    """
    return (nested_data['paths'],)
@pimms.calc('labels', 'label_weights')
def forward_labels(nested_data):
    """Retrieve the `'labels'` data from the `nested_data` initial processing.
    
    Parameters
    ----------
    nested_data : dict-like
        A dictionary of data from which the ouptuts are extracted. Nested data
        dictionaries are used to introduce lazy logic to the pipelines.

    Outputs
    -------
    labels : dict
        The label data derived from the drawn contours.
    """
    return (nested_data['labels'], nested_data['label_weights'])
fwdplan_contours = pimms.plan(
    constants=forward_constants,
    cortex=forward_cortex,
    flatmap=forward_flatmap,
    contours=forward_contours)
fwdplan_traces = pimms.plan(
    fwdplan_contours,
    traces=forward_traces)
fwdplan_paths = pimms.plan(
    fwdplan_traces,
    paths=forward_paths)
fwdplan_labels = pimms.plan(
    fwdplan_paths,
    labels=forward_labels)


# Loading/processing ----------------------------------------------------------
@pimms.calc('traces')
def calc_fwdtraces(rater, sid, chirality, save_path,
                   nested_data, io_options, region):
    """Either loads the trace data from the filesystem or retrieves it from the
    `nested_data` plan-data object.
    """
    h = chirality
    overwrite = io_options['overwrite']
    traces_path = os.path.join(save_path, 'traces')
    traces = None
    if overwrite is not True:
        # We try loading them and only calculate them if we aren't overwriting.
        try:
            traces = load_traces(rater, sid, h, region, load_path=traces_path)
        except FileNotFoundError:
            pass
    if traces is None:
        traces = nested_data['traces']
        # We don't try to save if overwrite is False because it will raise an
        # unnecessary error.
        if overwrite is not False:
            save_traces(
                rater, sid, h, traces,
                save_path=traces_path,
                filenames=region,
                **io_options)
    return (traces,)
@pimms.calc('paths')
def calc_paths(rater, sid, chirality, save_path,
               nested_data, cortex, io_options, region):
    """Either loads or calculates (and saves) then returns the paths from the
    (loaded or calculated) path-traces.
    """
    h = chirality
    overwrite = io_options['overwrite']
    paths_path = os.path.join(save_path, 'paths')
    paths = None
    if overwrite is not True:
        # We try loading them and only calculate them if we aren't overwriting.
        try:
            paths = load_paths(
                rater, sid, h, region,
                cortex=cortex,
                load_path=paths_path)
        except FileNotFoundError:
            pass
    if paths is None:
        paths = {}
        traces = nested_data['traces']
        for (name,trace) in traces.items():
            paths[name] = trace.to_path(cortex)
        if overwrite is not False:
            save_paths(
                rater, sid, h, paths,
                save_path=paths_path,
                filenames=region,
                **io_options)
    return (paths,)
@pimms.calc('labels', 'label_weights')
def calc_labels(rater, sid, chirality, save_path,
                nested_data, cortex, io_options, region,
                labelkey=labelkey):
    """Either loads or calculates (and saves) then returns the labels from the
    (loaded or calculated) paths.
    """
    h = chirality
    overwrite = io_options['overwrite']
    labels_path = os.path.join(save_path, 'labels')
    labels = None
    boundaries = procdata(region, 'boundaries')
    if overwrite is not True:
        # We try loading them and only calculate them if we aren't overwriting.
        try:
            ldat = load_labels(
                rater, sid, h, region,
                load_path=labels_path)
            return (ldat['labels'], ldat['weights'])
        except FileNotFoundError:
            pass
    if labels is None:
        weights = {}
        # Start by calculating the label weights from the paths.
        paths = nested_data['paths']
        for (name,path) in paths.items():
            if name not in boundaries:
                continue
            lbl = path.label
            if np.sum(lbl) > np.sum(1 - lbl):
                lbl = 1 - lbl
            weights[name] = lbl
        ws = np.array(list(weights.values()))
        # Add the zero label, which is the probability of not being in a
        # label.
        nolbl = np.max(
            [np.zeros(ws.shape[1]), 1 - np.sum(ws, axis=0)],
            axis=0)
        ws = np.concatenate([nolbl[None,:], ws])
        ws /= np.sum(ws, axis=0)
        # For the purpose of picking the labels, we don't want to label anything
        # as part of an area if the label value is less than or equal to 0.5.
        pick = np.array(ws)
        pp = pick[1:, :]
        pp[pp <= 0.5] = 0
        pick[1:, :] = pp
        # Here, we ensure that the argmax of a row without another label is 0.
        pick[0, :] = 0.25
        # Now find the argmax; 0 indicates none of the labels.
        lbl = np.argmax(pick, axis=0).astype(int)
        # Convert these indices into actual labels (per the label key).
        lblkey = np.array([0] + [labelkey[k] for k in weights.keys()])
        lbl = lblkey[lbl]
        ws = ws.T
        if overwrite is not False:
            ldat = {'labels': lbl, 'weights': ws}
            save_labels(
                rater, sid, h, ldat,
                save_path=labels_path,
                filenames=region,
                **io_options)
    return (lbl, ws)
@pimms.calc('reports')
def calc_reports(rater, sid, chirality, save_path,
                 nested_data, cortex, io_options, region,
                 labelkey=labelkey):
    """Either loads or calculates (and saves) then returns the surface area reports
    from the (loaded or calculated) labels.
    """
    h = chirality
    overwrite = io_options['overwrite']
    reports_path = os.path.join(save_path, 'reports')
    reports = None
    if overwrite is not True:
        # We try loading them and only calculate them if we aren't overwriting.
        try:
            reports = load_reports(
                rater, sid, h, region,
                load_path=reports_path)
        except FileNotFoundError:
            pass
    if reports is None:
        # Calculate them from the cortex surface area and save them.
        hem = nested_data['cortex']
        # We need the roi label for this.
        rlabelkey = {v:k for (k,v) in labelkey.items()}
        roi = hem.prop('cortex_label')
        sa = hem.prop('midgray_surface_area')
        cortex_sa = np.sum(sa[roi])
        lbl = nested_data['labels']
        lbl_sareas = {'cortex': cortex_sa}
        for ll in np.unique(lbl):
            if ll == 0:
                continue
            ll_sa = sa[lbl == ll]
            name = rlabelkey[ll]
            lbl_sareas[name] = np.sum(ll_sa)
        reports = {'surface_area': lbl_sareas}
        # Save the reports
        if overwrite is not False:
            save_reports(
                rater, sid, h, reports,
                save_path=reports_path,
                filenames=region,
                **io_options)
    return (reports,)


# Plans ------------------------------------------------------------------------
traces_plan = pimms.plan(
    fwdplan_contours,
    traces=calc_fwdtraces)
paths_plan = pimms.plan(
    fwdplan_traces,
    paths=calc_paths)
labels_plan = pimms.plan(
    fwdplan_paths,
    labels=calc_labels)
reports_plan = pimms.plan(
    fwdplan_labels,
    reports=calc_reports)


# Means ########################################################################

def export_means(sid, h,
                 save_path='.',
                 load_path=None,
                 raters=None,
                 npoints=500,
                 overwrite=True,
                 mkdir=True,
                 mkdir_mode=0o775,
                 vc_contours='ventral_meanrater',
                 vc_input_traces='ventral',
                 min_raters=1,
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
        The default is `hcpannot.analysis.vc_contours_ventral_meanrater`.
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
    if isinstance(vc_contours, str):
        if vc_contours == 'ventral':
            vc_contours = vc_contours_ventral
        elif vc_contours == 'ventral_meanrater':
            vc_contours = vc_contours_ventral_meanrater
        else:
            raise ValueError(f"unrecognized contours: {vc_contours}")
    if isinstance(vc_input_traces, str):
        if vc_input_traces in ('ventral', 'ventral_meanrater'):
            from .analysis import vc_meanrater_input_traces_ventral as \
                vc_input_traces
        else:
            raise ValueError(f"unrecognized mean traces: {vc_input_traces}")
    # This is where we will load and/or eventually save these contour files.
    data_path = to_data_path(meanrater, sid, save_path=save_path)
    # First, check if these data already exist (if we're not overwriting).
    if not overwrite and os.path.isdir(data_path):
        try:
            # We use save_path here because we are checking for complete results
            # that have already been calculated and saved.
            return load_contours(
                meanrater, sid, h, save_path,
                vc_contours=vc_contours,
                error_on_missing=True)
        except Exception:
            pass
    # First things first: we need to load in the traces of all raters.
    if raters is None:
        raters = guess_raters(load_path)
    trs = {}
    for rater in raters:
        # Try loading the traces.
        try:
            tr = load_traces(rater, sid, h, load_path,
                             vc_traces=vc_input_traces)
        except Exception as e:
            print(e)
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
        if len(trlist) < min_raters:
            m = f"too few raters for contour {k}: {len(trlist)} of {min_raters}"
            raise ValueError(m)
        meantrs[k] = np.mean([u[k].points for u in trlist], axis=0)
        rcounts[k] = len(trlist)
        # For some reason the LH is always in the wrong direction here.
        if h == 'lh':
            meantrs[k] = np.fliplr(meantrs[k])
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


# Images #######################################################################

def export_images(rater, sid, h,
                  save_path='.',
                  contours_load_path=None,
                  labels_load_path=None,
                  vc_plan='ventral',
                  vc_contours='ventral',
                  overwrite=True,
                  mkdir=True,
                  mkdir_mode=0o775,
                  figsize=4,
                  dpi=1024):
    """Exports an image of the labels and contours for a set of  visual areas.

    Exports the requested images to disk. The images include the original
    contours and are colored according to the labels.

    Parameters
    ----------
    rater : str 
        The rater to export the labels for.
    sid : int
        The HCP subject ID of the subject whose contours should be processed.
    h : str
        The hemisphere, `'lh'` or `'rh'`.
    save_path : directory name, optional
        The directory to which this set of labels should be saved. Labels
        themselves are saved into a directory equivalen to
        `os.path.join(save_path, rater)`. The default is `'.'`.
    contours_load_path : directory name, optional
        The directory from which contours should be loaded; if not provided, 
        then defaults to the `save_path`.
    labels_load_path : directory name, optional
        The directory from which labels should be loaded; if not provided, 
        then defaults to the `save_path`.
    vc_plan : pimms calculation plan, optional
        The plan that is to be executed on the contours. This plan must produce
        an output value called `'traces'` that contains the traces to be saved
        to disk. The default is `hcpannot.vc_plan`.
    vc_contours : dict
        A dictionary whose keys are the names of the contours and whose values
        are format strings for the filenames that store the given contours. The
        contours correspond to those drawn by raters using the annotation tool.
        The default is `hcpannot.analysis.vc_contours`.
    """
    # If load path isn't provided, we guess it.
    if contours_load_path is None:
        contours_load_path = save_path
    if labels_load_path is None:
        labels_load_path = save_path
    if isinstance(vc_contours, str):
        tmp = vc_contours_by_name.get(vc_contours)
        if tmp is None:
            raise ValueError(f"unrecognized contours: {vc_contours}")
        vc_contours = tmp
    if isinstance(vc_plan, str):
        tmp = vc_plan_by_name.get(vc_plan)
        if tmp is None:
            raise ValueError(f"unrecognized plan: {vc_plan}")
        vc_plan = tmp
    if not isinstance(figsize, Sequence):
        figsize = (figsize, figsize)
    # Make sure we need to generate the images in the first place.
    path = os.path.join(save_path, rater)
    filename = os.path.join(path, f'{h}_{sid}.png')
    if not overwrite and os.path.isfile(filename):
        return filename
    if mkdir and not os.path.isdir(path):
        os.makedirs(path, mode=mkdir_mode, exist_ok=True)
    # Load the labels:
    lbl = load_labels(rater, sid, h, labels_load_path)
    # The easiest way to load the contours and the flatmap is to use the
    # vc_plan, so we do:
    dat = vc_plan(rater=rater, sid=sid, hemisphere=h,
                  save_path=contours_load_path,
                  vc_contours=vc_contours)
    contours = dat['contours']
    flatmap = dat['flatmap']
    # Now we can setup the figure.
    import matplotlib.pyplot as plt
    (fig,axs) = plt.subplots(2,2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(0,0,1,1,0,0)
    # We plot the contours on each plot.
    contour_colors = {
        k: f'{v:3.2f}'
        for (k,v) in zip(
            sorted(contours.keys()),
            np.linspace(0, 1, len(contours)+1)[1:])}
    v123 = subject_data[(sid,h)]['v123']
    for ax in axs.flatten():
        # V123 first:
        for (k,(x,y)) in v123.items():
            ax.plot(x, y, 'k-', lw=0.5, zorder=9)
        # Then the contours we care about.
        for (k,(x,y)) in contours.items():
            ax.plot(x, y, '-', lw=1, c=contour_colors[k], zorder=10)
        ax.axis('off')
    # Now plot the appropriate flatmaps
    lbl = lbl[flatmap.labels]
    ny.cortex_plot(flatmap, axes=axs[0,0], color=lbl, mask=(lbl>0), cmap='hsv')
    ny.cortex_plot(flatmap, axes=axs[1,0])
    mask = ('prf_variance_explained', 0.1, 1)
    ny.cortex_plot(flatmap, axes=axs[0,1], color='prf_polar_angle', mask=mask)
    ny.cortex_plot(flatmap, axes=axs[1,1], color='prf_eccentricity', mask=mask)
    # Zoom in on the upper right plot.
    (xmin, xmax) = (np.inf, -np.inf)
    (ymin, ymax) = (np.inf, -np.inf)
    for coords in contours.values():
        (xmn,ymn) = np.min(coords, axis=1)
        (xmx,ymx) = np.max(coords, axis=1)
        xmin = min(xmin, xmn)
        xmax = max(xmax, xmx)
        ymin = min(ymin, ymn)
        ymax = max(ymax, ymx)
    xmu = (xmin + xmax) / 2
    ymu = (ymin + ymax) / 2
    dx = 1.5*(xmax - xmu)
    dy = 1.5*(ymax - ymu)
    axs[0,0].set_xlim([xmu - dx, xmu + dx])
    axs[0,0].set_ylim([ymu - dy, ymu + dy])
    # The figure is finished now, so save it and close it.
    plt.savefig(filename)
    plt.close(fig)
    return filename


# The below is archived code on processing the traces using the watershed
# algorithm, which works sometimes but not consistently.
def export_watershed_data(raters, sid, h,
                          vc_plan='ventral',
                          save_path='.',
                          load_path=None,
                          overwrite=True,
                          mkdir=True,
                          mkdir_mode=0o775,
                          plan_args=None):
    """Exports the labels for the visual areas hV4, VO1, and VO2.

    Parameters
    ----------
    raters : str or list of str
        The rater or raters to export the labels for.
    sid : int
        The HCP subject ID of the subject whose contours should be processed.
    h : 'lh' or 'rh'
        The hemisphere that should be processed.
    vc_plan : pimms plan
        The calculation plan that is to be used to produce the labels and
        surface areas. These values must be stored in the `'labels'` and
        `'surface_areas'` elements of the resulting plan dictionary. The plan
        should resemble the `hcpannot.analysis.vc_plan_ventral` plan.
    save_path : directory name, optional
        The directory to which this set of labels should be saved. Labels
        themselves are saved into a directory equivalen to
        `os.path.join(save_path, rater)`. The default is `'.'`.
    load_path : directory name, optional
        The directory from which traces should be loaded; if not provided, then
        defaults to the `save_path`.
    plan_args : dict
        The arguments that should be passed to the `vc_plan`. If `None` (the
        default) then `{}` is used. These arguments are merged into a dictionary
        containing the `rater`, `sid`, and `hemisphere` options, but arguments
        in the `plan_args` dictionary overwrite these arguments if provided.
    """
    from os.path import isfile
    # If load path isn't provided, we guess it.
    if load_path is None:
        load_path = save_path
    # We'll be iterating through the raters, so make sure it's a sequence.
    if isinstance(raters, str):
        raters = [raters]
    if plan_args is None:
        plan_args = {}
    if load_path is None:
        load_path = save_path
    if isinstance(vc_plan, str):
        if vc_plan == 'ventral':
            from .analysis import vc_plan_ventral as vc_plan
        else:
            raise ValueError(f"unrecognized vc_plan: {vc_plan}")
    # Now iterate through the raters and hemispheres.
    saved = []
    for rater in raters:
        # First of all, if there are already labels, don't rerun this unless
        # the plan is to overwrite things.
        path = os.path.join(save_path, rater)
        if mkdir and not os.path.exists(path):
            os.makedirs(path, mkdir_mode, exist_ok=True)
        lbls_filename = os.path.join(path, f'{h}_{sid}.mgz')
        sarea_filename = os.path.join(path, f'{h}_{sid}_sareas.tsv')
        if not overwrite:
            if isfile(lbls_filename) and isfile(sarea_filename):
                saved.append(lbls_filename)
                saved.append(sarea_filename)
                continue
        # We need to calculate the labels, so set up the plan:
        args = dict(rater=rater, sid=sid, hemisphere=h, save_path=load_path)
        args.update(**plan_args)
        data = vc_plan(**args)
        # Now we just extract and save the labels out to disk.
        ny.save(lbls_filename, data['labels'])
        # And finally, save the tsv file:
        with open(sarea_filename, 'wt') as fl:
            print("ROI\tsurface_area", file=fl)
            for (k,v) in data['surface_areas'].items():
                print(f"{k}\t{v}\n", file=fl)
        # That's all!
        saved.append(lbls_filename)
        saved.append(sarea_filename)
    return saved
