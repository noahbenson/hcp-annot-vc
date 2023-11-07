################################################################################
# proc/__init__.py
#
# Initialization module for the proc namespace, which stores the processing code
# for the various regions annotated in the HCP annotation project.

from .core    import init_plan
from .ventral import ventral_contours_plan, ventral_contours_meanplan

contours_plans = {
    'ventral': ventral_contours_plan,
    'meanventral': ventral_contours_meanplan}

def proc(contours_plan, **kw):
    """Returns a visual cortex processing plan dict based on the given subplan.

    The given subplan should typically be the name of a known plan, such as
    `'ventral'` for the ventral processing plan. If it is a plan object itself,
    then it must include the following keys at a minimum: `'rater'`, `'sid'`,
    `'chirality'`, `'hemisphere'`, `'io_options'`, `'cortex'`, `'flatmap'`, and
    `'traces'`.

    Typically such a plan is created by merging calculations with the
    `init_plan`. The following parameters and outputs should be universal to all
    such correctly-created plans. The return value of the `proc` function is a
    dictionary containing all of the parameters and outputs listed below as
    keys.   

    Parameters
    ----------
    contours_plan : pimms plan or str
        A pimms plan object or a string that names such an object. Contour plans
        with names are stored in the dictionary `hcpannot.proc.contours_plans`.
        The contour plan must produce an output `'traces'` that is a dictionary
        of the neuropythy path-trace objects (see the `'traces'` output below).
    region : str, optional
        The region name for the contours being loaded. Although this parameter
        is an optional named parameter, it is required by the various plans and
        so must be provided. If `region` is not provided but the `contours_plan`
        is a string, then the `contours_plan` string is used as the region.
    rater : str, optional
        The rater whose contours are to be loaded. Although this parameter is
        an optional named parameter, it is required by the various plans and so
        must be provided.
    sid : int, optional
        The HCP subject-ID of the subject to load. Although this parameter is
        an optional named parameter, it is required by the various plans and so
        must be provided.
    load_path : str, optional
        The path of the `save/` directory from the `hcp-annot-vc:data`
        repository from which contours are loaded. Although this parameter is an
        optional named parameter, it is required by the various plans and so
        must be provided.
    save_path : str, optional
        The path into which the various processed data are saved. This should be
        a directory, and the data are saved into subdirectories named `traces`,
        `paths`, `labels`, etc. Although this parameter is an optional named
        parameter, it is required by the various plans and so must be provided.
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
    labelkey : dict, optional
        The key that maps label names (such as `'hV4'`) to integer label values.
        By default the `hcpannot.config.labelkey` dictionary is used.

    Outputs
    -------
    chirality : 'lh' or 'rh'
        The chirality of the hemisphere parameter; this is included in case the
        hemisphere is something like `'lh_LR32k'`.
    cortex : neuropythy Cortex object
        A neuropythy cortex object representing the hemisphere being processed.
    flatmap : neuropythy Mesh object
        A 2D flatmap of the occipital pole of the `cortex` object whose contours
        are being processed.
    traces : dict
        A dictionary whose keys are the names of the traces processed by the
        contours plan and whose values are neuropythy `PathTrace` objects
        representing those traces.
    paths : dict
        A dictionary whose keys are the names of the paths processed by the
        contours plan and whose values are neuropythy `Path` objects
        representing those paths. Unlike traces, paths are registered to a
        specific cortical surface (`cortex` in this case).
    labels : numpy vector
        A vector of integer values corresponding to the labels of each vertex on
        the cortical surface (`cortex`). The specific label values are derived
        from the `labelkey` parameter.
    label_weights : numpy matrix
        A matrix with one row per cortical surface vertex. The columns
        correspond to the different potential areas and give a relative weight
        to each area.
    reports : dict
        A dictionary of reports generated (typically surface area).

    """
    from .core import traces_plan, paths_plan, labels_plan, reports_plan
    if isinstance(contours_plan, str):
        cp = contours_plans.get(contours_plan)
        if cp is None:
            raise ValueError(f"unrecognized contours plan: {contours_plan}")
        if 'region' not in kw:
            kw['region'] = contours_plan
        contours_plan = cp
    save_path = kw.pop('save_path')
    contours_data = contours_plan(**kw)
    traces_data = traces_plan(nested_data=contours_data, save_path=save_path)
    paths_data = paths_plan(nested_data=traces_data, save_path=save_path)
    labels_data = labels_plan(nested_data=paths_data, save_path=save_path)
    reports_data = reports_plan(nested_data=labels_data, save_path=save_path)
    return reports_data
def proc_all(contours_plan, **kw):
    """Processes one or all of the given plans and returns a dataframe.

    Repeatedly calls `proc` on the raters, subjects, and hemispheres given in
    the keyword arguments. If errors arise, they are logged. Otherwise, the
    returned dataframe contains meta-data about the processing.

    For information on parameters and outputs, see the `proc` function.
    """
    from time import time
    from pandas import DataFrame
    from numbers import Integral
    raters = kw.pop('rater')
    sids = kw.pop('sid')
    hs = kw.pop('hemisphere')
    if isinstance(raters, str):
        raters = [raters]
    if isinstance(sids, Integral):
        sids = [sids]
    if isinstance(hs, str):
        hs = [hs]
    # We process by subject and hemisphere first because it is smarter in terms
    # of how/when we do disk i/o.
    res = dict(rater=[], sid=[], hemisphere=[], dt=[], error=[])
    for sid in sids:
        for h in hs:
            for rater in raters:
                error = ''
                t0 = time()
                try:
                    data = proc(
                        contours_plan,
                        rater=rater, sid=sid, hemisphere=h,
                        **kw)
                    # Force all the data to calculate.
                    traces = data['traces']
                    paths = data['paths']
                    labels = data['labels']
                    reports = data['reports']
                except Exception as e:
                    error = str(e)
                t1 = time()
                # That's all.
                res['rater'].append(rater)
                res['sid'].append(sid)
                res['hemisphere'].append(h)
                res['dt'].append(t1 - t0)
                res['error'].append(error)
    return DataFrame(res)
def meanproc(contours_plan, **kw):
    """Returns a processing plan dictionary for the mean contours.

    The `meanproc` function is roughly equivalent to the `proc` function except
    that it operates over the processed traces of a set of raters and produces
    contours that represent the mean of these traces then processes these
    contours. The `meanproc` function has a small number of differences from the
    `proc` function, which are listed here.
     * The `rater` parameter of `meanproc` is optional, and the default rater
       name is `'mean'`.
     * The parameter `source_raters` may be given a list of raters whose traces
       should be averaged. By default (or if `None` is given), all of the raters
       defined in the `hcpannot.config` namespace for the given `region` are
       used.
     * If the `contours_plan` argument is a string like `'ventral'` that does
       not start with `'mean'`, then `'mean'` is prepended to it. The same is
       true of the `region` parameter.
     * The `load_path` option, if not provided, defaults to the subdirectory
       `traces` of the `save_path`, under the assumption that the processing
       for the individual raters and for the mean contours are using the same
       output directory.
    """
    from ..config import (procdata, meanrater)
    from .core import traces_plan, paths_plan, labels_plan, reports_plan
    # Process the arguments.
    rater = kw.pop('rater', meanrater)
    kw['rater'] = rater
    if isinstance(contours_plan, str):
        if not contours_plan.startswith('mean'):
            contours_plan = 'mean' + contours_plan
        cp = contours_plans.get(contours_plan)
        if cp is None:
            raise ValueError(f"unrecognized contours plan: {contours_plan}")
        if 'region' not in kw:
            kw['region'] = contours_plan
        contours_plan = cp
    reg = kw.get('region', None)
    if isinstance(reg, str) and not reg.startswith('mean'):
        kw['region'] = 'mean' + reg
    save_path = kw['save_path']
    load_path = kw.get('load_path', None)
    if load_path is None:
        kw['load_path'] = save_path
    # Run and nest the plans.
    contours_data = contours_plan(**kw)
    traces_data = traces_plan(nested_data=contours_data, save_path=save_path)
    paths_data = paths_plan(nested_data=traces_data, save_path=save_path)
    labels_data = labels_plan(nested_data=paths_data, save_path=save_path)
    reports_data = reports_plan(nested_data=labels_data, save_path=save_path)
    return reports_data
def meanproc_all(contours_plan, **kw):
    """Processes one or all of the given mean plans and returns a dataframe.

    Repeatedly calls `meanproc` on the subjects and hemispheres given in the
    keyword arguments. If errors arise, they are logged. Otherwise, the returned
    dataframe contains meta-data about the processing.

    For information on parameters and outputs, see the `meanproc` function.
    """
    from time import time
    from pandas import DataFrame
    from numbers import Integral
    from ..config import meanrater
    meanrater = kw.get('rater', meanrater)
    sids = kw.pop('sid')
    hs = kw.pop('hemisphere')
    if isinstance(sids, Integral):
        sids = [sids]
    if isinstance(hs, str):
        hs = [hs]
    # We process by subject and hemisphere first because it is smarter in terms
    # of how/when we do disk i/o.
    res = dict(rater=[], sid=[], hemisphere=[], dt=[], error=[])
    for sid in sids:
        for h in hs:
            error = ''
            t0 = time()
            try:
                data = meanproc(
                    contours_plan,
                    sid=sid, hemisphere=h,
                    **kw)
                # Force all the data to calculate.
                traces = data['traces']
                paths = data['paths']
                labels = data['labels']
                reports = data['reports']
            except Exception as e:
                error = str(e)
            t1 = time()
            # Note that we fill a 'rater' column with 'mean' to be
            # consisitent with the proc_all function outputs.
            res['rater'].append(meanrater)
            res['sid'].append(sid)
            res['hemisphere'].append(h)
            res['dt'].append(t1 - t0)
            res['error'].append(error)
    return DataFrame(res)
