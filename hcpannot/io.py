################################################################################
# io.py
#
# Input and output tools for the HCP visual cortex contours.
# by Noah C. Benson <nben@uw.edu>


# Dependencies #################################################################

import os, json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import neuropythy as ny

from .config import (
    procdata,
    to_data_path)


# Utilities ####################################################################

def guess_raters(path):
    """Returns a list of possible rater names in the given path.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    return [
        flnm for flnm in os.listdir(path)
        if not flnm.startswith('.')
        if os.path.isdir(os.path.join(path, flnm))]
def _save_annotdata(tag, savefn, rater, sid, h, annots,
                    save_path=None, filenames=None, default_filename=None,
                    overwrite=True, mkdir=True, mkdir_mode=0o775,
                    step=None):
    """Saves a set of data for a rater, subject, and hemisphere.

    The `save_annotdata` function is meant as a helper function for the more
    specific functions `save_contours`, `save_traces`, `save_paths`, etc. It
    requires most of the same arguments as these functions in addition to the
    `tag` and `savefn` parameters. The `tag` is the name of the data being
    saved (such as `"contours"`) while the `savefn` is a function that, given
    a valid filename and the data to be saved, actually performs the save.
    """
    import os, neuropythy as ny
    if step is None:
        step = tag
    if isinstance(filenames, str):
        filenames = procdata(filenames, step)
    elif filenames is None and default_filename is not None:
        filenames = {None: default_filename}
    elif not isinstance(filenames, Mapping):
        raise ValueError(
            f"save_{tag} [{rater}/{sid}/{h}] argument filenames has invalid"
            f" type {type(filenames)} (must be str, None, or dict-like)")
    if not isinstance(annots, Mapping):
        raise ValueError(
            f"save_{tag} [{rater}/{sid}/{h}] argument contours has invalid"
            f" type {type(annots)} (must be dict-like)")
    if overwrite not in (True, False, None):
        raise ValueError(
            f"save_{tag}s [{rater}/{sid}/{h}] argument overwrite must be a"
            f" boolean value or None")
    # Put together the directory and make sure it exists.
    data_path = to_data_path(
        rater, sid, save_path,
        mkdir=mkdir,
        mkdir_mode=mkdir_mode)
    # Make sure the hemisphere is okay.
    h = ny.to_hemi_str(h)
    # Now, step through the contours, saving out the ones that have entries
    # in the filenames dict.
    saved_fls = {}
    for (name, annot) in annots.items():
        # Figure out the filename that this contour gets saved into.
        filename_tmpl = filenames.get(name, None)
        if filename_tmpl is None:
            filename_tmpl = filenames.get(None, None)
        if filename_tmpl is None:
            raise ValueError(
                f"save_{tag} [{rater}/{sid}/{h}] could not find filename"
                f" template for {name}")
        filename = filename_tmpl.format(
            rater=rater,
            sid=sid,
            hemisphere=h,
            name=name)
        filename = data_path / filename
        # See if we are overwriting this file or not.
        exists = filename.exists()
        if overwrite is False and exists:
            raise RuntimeError(
                f"save_{tag}s [{rater}/{sid}/{h}] filename exists and"
                f" overwrite is False: {filename}")
        elif overwrite is True or not exists:
            # Actually save the file out to disk.
            try:
                savefn(filename, annot)
            except Exception as e:
                raise RuntimeError(
                    f"save_{tag} [{rater}/{sid}/{h}] failed for {name}:"
                    f" ({type(e)}) {e}")
            # Add this file to the results list only if it was actually saved.
            saved_fls[name] = filename
    return saved_fls
def _load_annotdata(tag, loadfn, rater, sid, h, filenames,
                    load_path=None, missing_okay=False, step=None):
    """Loads a set of annotation data for a rater, subject, and hemisphere.
    
    This function is used by the _load_contours, _load_traces, load_paths, etc.
    functions below to standardize the loading of processing data. It requires
    most of the same arguments as these functions in addition to the `tag` and
    `loadfn` parameters. The `tag` is the name of the data being saved (such as
    `"contours"`) while the `loadfn` is a function that, given a valid filename
    and the data to be loaded, actually loads and returns the data.
    """
    import os, neuropythy as ny
    if step is None:
        step = tag
    if isinstance(filenames, str):
        filenames = procdata(filenames, step)
    elif not isinstance(filenames, Mapping):
        raise ValueError(
            f"load_{tag} [{rater}/{sid}/{h}] argument filenames has invalid"
            f" type {type(filenames)} (must be str or dict-like)")
    # Put together the directory and make sure it exists.
    data_path = to_data_path(rater, sid, load_path, mkdir=False)
    # Make sure the hemisphere is okay.
    h = ny.to_hemi_str(h)
    # We need to load precisely the files in filenames:
    annots = {}
    for (name,filename_tmpl) in filenames.items():
        filename = filename_tmpl.format(
            rater=rater,
            sid=sid,
            hemisphere=h,
            name=name)
        filepath = data_path / filename
        # Run the load function.
        if filepath.exists():
            try:
                annots[name] = loadfn(filepath)
            except Exception as e:
                raise RuntimeError(
                    f"load_{tag} [{rater}/{sid}/{h}] failed for {name}:"
                    f" ({type(e)}) {e}")
        elif not missing_okay:
            raise FileNotFoundError(
                f"load_{tag} [{rater}/{sid}/{h}] file not found for {name}:"
                f" {str(filepath)}")
    return annots


# Contours #####################################################################

def _save_contour(filepath, contour):
    "Saves a single contour to the given filename."
    # Check the contour coordinates.
    from .interface import flatmap_to_imgrid
    if ny.is_path_trace(contour):
        coords = contour.points
    else:
        coords = np.asarray(contour)
    if len(coords.shape) != 2 or coords.shape[0] != 2:
        raise ValueError(f"shape {coords.shape} is invalid")
    coords = flatmap_to_imgrid(coords)[0,0]
    # In the json file, coordinates are always stored in N x 2 matrices.
    coords = coords.T
    # Actually save the file out to disk.
    filepath = Path(filepath)
    with filepath.open('wt') as fl:
        json.dump(coords.tolist(), fl)
    return filepath
def _load_contour(filepath):
    "Loads a single contour from the given filepath."
    from .interface import imgrid_to_flatmap
    # Read in the json file.
    with Path(filepath).open('rt') as file:
        contour = np.array(json.load(file))
    # The coordinates are stored in an N x 2 matrix in the json file.
    contour = contour.T
    # Convert from image-grid coordinates into flatmap coordinates. 
    return imgrid_to_flatmap(contour)
def save_contours(rater, sid, h, contours,
                  save_path=None, filenames=None,
                  overwrite=True, mkdir=True, mkdir_mode=0o775):
    """Saves a set of contours for a subject and hemisphere to a save path.

    `save_contours(rater, sid, h, contours, save_path)` saves a set of contours
    (`contours`) to the given `save_path` for the given rater, subject ID
    (`sid`) and hemisphere (`h`). The return value is a dict whose keys are the
    contour names and whose values are the filenames to which the contour was
    saved.

    Parameters
    ----------
    rater : str or None
        The name of the rater into whose directory these annotationss should be
        saved. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being saved. If
        `None` is given, then the subject id is not included in the output
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        saved.
    contours : dict
        A dictionary whose keys are the names of the contours to be saved and
        whose values are the contours themselves (which must be numpy arrays).
    save_path : path-like or None
        The directory into which the annotationss are to be saved. Note that
        this directory is the base directory that is combined with the rater and
        sid to produce an actual save directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the annotationss that are to
        be saved and whose values are the filename templates. If this option is
        excluded or provided the value `None`, then all of the annotations in
        `contours` are saved and the filename template is
        `'{hemisphere}.{name}.json'`. If `filenames` is a string, then it is
        looked up in the `'contours'` step of the processing data using the
        `procdata` function (e.g. if `filenames == 'ventral'` then the dict
        `procdata('ventral', 'contours')` is used. If the filenames dict
        contains an entry whose key is `None`, then that entry is used as a
        catch-all for contours whose names are not explicitly listed. The
        filename template(s) may use the `{hemisphere}`, `{sid}`, `{rater}`, and
        `{name}` format ids.
    overwrite : boolean or None, optional
        Whether to overwrite files if they already exist. If `False` is given,
        then existing annotation files are not ovewritten and an exception is
        raised if an existing annotation file is found. If `True` is given (the
        default) then existing files are overwritten. If `None` is given, then
        existing annotation files are not overwritten, but no errors are raised
        if they are found.
    mkdir : boolean, optional
        Whether to create the path directory if it does not exist. If `True` is
        given (the default), then all parent directories and the requested
        directory are created with the mode given by `mkdir_mode`. If `False` is
        given, then no directories are created.
    mkdir_mode : int, optional
        The mode to be used when creating directories. The default is `0o775`.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` parameter and
        whose values are the paths of the files that were successfully saved.
    """
    return _save_annotdata(
        'contours', _save_contour,
        rater, sid, h, contours,
        save_path=save_path,
        filenames=filenames,
        default_filename='{hemisphere}.{name}.json',
        overwrite=overwrite,
        mkdir=mkdir,
        mkdir_mode=mkdir_mode)
def load_contours(rater, sid, h, filenames,
                  load_path=None,
                  missing_okay=False):
    """Loads a set of contours for a subject and hemisphere from a save path.

    `load_contours(rater, sid, h, filenames, load_path)` loads a set of contours
    from the given `load_path` for the given rater, subject ID (`sid`) and
    hemisphere (`h`). These contours are returned as a dict whose keys are the
    contours names and whose values are the contour points. The contour names
    and filenames are provided in the `filenames` argument which must be a
    dictionary whose keys are contour names and whose values are filename
    templates (see below).

    Parameters
    ----------
    rater : str or None
        The name of the rater from whose directory these annotationss should be
        loaded. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being loaded. If
        `None` is given, then the subject id is not included in the input
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        loaded.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the annotationss that are to
        be loaded and whose values are the filename templates. If `filenames` is
        a string, then it is looked up in the `'contours'` step of the
        processing data using the `procdata` function (e.g. if `filenames ==
        'ventral'` then the dict `procdata('ventral', 'contours')` is used.
    load_path : path-like or None
        The directory from which the annotationss are to be loaded. Note that
        this directory is the base directory that is combined with the rater and
        sid to produce an actual load directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    missing_okay : boolean, optional
        Whether it is okay for some or all of the files in `filenames` to be missing.
        If `True` then errors are not raised when a missing file is encountered and
        instead no entry in the resulting dictionary is included. If `False` (the
        default), then an error is raised if a file is missing.
    """
    return _load_annotdata(
        'contours', _load_contour,
        rater, sid, h, filenames,
        load_path=load_path,
        missing_okay=missing_okay)


# Traces #######################################################################

def _save_trace(filepath, trace):
    "Save a single trace to a filename."
    # We need to clear the mesh field for this to work properly.
    mp = trace.map_projection
    if mp is None:
        tr = trace
    else:
        mp = mp.copy(mesh=None)
        tr = trace.copy(map_projection=mp)
    ny.save(os.fspath(filepath), ny.util.normalize(tr.normalize()), 'json')
    return filepath
def _load_trace(filepath, mesh=None):
    "Loads a single trace from the given filepath."
    trace = ny.load(os.fspath(filepath), 'json')
    if not isinstance(trace, dict):
        raise RuntimeError("file does not contain a json mapping")
    mpj = trace.pop('map_projection')
    pts = trace.pop('points')
    if mesh is not None:
        mpj = mpj.copy(mesh=mesh)
    trace = ny.geometry.PathTrace(mpj, pts, **trace)
    return trace
def save_traces(rater, sid, h, traces,
                save_path=None, filenames=None,
                overwrite=True, mkdir=True, mkdir_mode=0o775):
    """Saves a set of traces for a subject and hemisphere to a save path.

    `save_traces(rater, sid, h, contours, save_path)` saves a set of contour
    traces (`traces`) to the given `save_path` for the given rater, subject ID
    (`sid`) and hemisphere (`h`). The return value is a dict whose keys are the
    trace names and whose values are the filenames to which the trace was saved.

    Parameters
    ----------
    rater : str or None
        The name of the rater into whose directory the annotations should be
        saved. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being saved. If
        `None` is given, then the subject id is not included in the output
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        saved.
    traces : dict
        A dictionary whose keys are the names of the traces to be saved and
        whose values are the traces themselves (which must be neuropythy
        `PathTrace` objects).
    save_path : path-like or None
        The directory into which the annotations are to be saved. Note that this
        directory is the base directory that is combined with the rater and
        sid to produce an actual save directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the traces that are to be
        saved and whose values are the filename templates. If this option is
        excluded or provided the value `None`, then all of the traces in
        `traces` are saved and the filename template is
        `'{hemisphere}.{name}_trace.json.gz'`. If `filenames` is a string, then
        it is looked up in the `'traces'` step of the processing data using the
        `procdata` function (e.g. if `filenames == 'ventral'` then the dict
        `procdata('ventral', 'traces')` is used. If the filenames dict contains
        an entry whose key is `None`, then that entry is used as a catch-all for
        traces whose names are not explicitly listed. The filename template(s)
        may use the `{hemisphere}`, `{sid}`, `{rater}`, and `{name}` format ids.
    overwrite : boolean or None, optional
        Whether to overwrite files if they already exist. If `False` is given,
        then existing files are not ovewritten and an exception is raised if an
        existing file is found. If `True` is given (the default) then existing
        files are overwritten. If `None` is given, then existing contour files
        are not overwritten, but no errors are raised if they are found.
    mkdir : boolean, optional
        Whether to create the path directory if it does not exist. If `True` is
        given (the default), then all parent directories and the requested
        directory are created with the mode given by `mkdir_mode`. If `False` is
        given, then no directories are created.
    mkdir_mode : int, optional
        The mode to be used when creating directories. The default is `0o775`.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` parameter and
        whose values are the paths of the files that were successfully saved.
    """
    return _save_annotdata(
        'traces', _save_trace,
        rater, sid, h, traces,
        save_path=save_path,
        filenames=filenames,
        default_filename='{hemisphere}.{name}_trace.json.gz',
        overwrite=overwrite,
        mkdir=mkdir,
        mkdir_mode=mkdir_mode)
def load_traces(rater, sid, h, filenames,
                load_path=None,
                missing_okay=False):
    """Loads a set of traces for a subject and hemisphere from a save path.

    `load_traces(rater, sid, h, filenames, load_path)` loads a set of contours
    from the given `load_path` for the given rater, subject ID (`sid`) and
    hemisphere (`h`). These traces are returned as a dict whose keys are the
    traces names and whose values are the neuropythy `PathTrace` objects. The
    trace names and filenames are provided in the `filenames` argument which
    must be a dictionary whose keys are trace names and whose values are
    filename templates (see below).

    Parameters
    ----------
    rater : str or None
        The name of the rater from whose directory these annotationss should be
        loaded. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being loaded. If
        `None` is given, then the subject id is not included in the input
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        loaded.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the annotationss that are to
        be loaded and whose values are the filename templates. If `filenames` is
        a string, then it is looked up in the `'traces'` step of the
        processing data using the `procdata` function (e.g. if `filenames ==
        'ventral'` then the dict `procdata('ventral', 'traces')` is used.
    load_path : path-like or None
        The directory from which the annotationss are to be loaded. Note that
        this directory is the base directory that is combined with the rater and
        sid to produce an actual load directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    missing_okay : boolean, optional
        Whether it is okay for some or all of the files in `filenames` to be missing.
        If `True` then errors are not raised when a missing file is encountered and
        instead no entry in the resulting dictionary is included. If `False` (the
        default), then an error is raised if a file is missing.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` argument and
        whose values are the neuropythy `PathTrace` objects that were 
        successfully loaded.
    """
    return _load_annotdata(
        'traces', _load_trace,
        rater, sid, h, filenames,
        load_path=load_path,
        missing_okay=missing_okay)


# Paths ########################################################################

def _save_path(filepath, path):
    "Save a single path to a filename."
    # We need to clear the mesh field for this to work properly.
    pathdata = dict(path.addresses)
    ny.save(os.fspath(filepath), pathdata)
    return filepath
def _load_path(filepath, hem):
    "Loads a single path from the given filepath."
    pathdata = ny.load(os.fspath(filepath))
    faces = np.asarray(pathdata['faces'])
    barys = np.asarray(pathdata['coordinates'])
    addr = {'faces': faces, 'coordinates': barys}
    return ny.geometry.Path(hem, addr)
def save_paths(rater, sid, h, paths,
               save_path=None, filenames=None,
               overwrite=True, mkdir=True, mkdir_mode=0o775):
    """Saves a set of paths for a subject and hemisphere to a save path.

    `save_paths(rater, sid, h, contours, save_path)` saves a set of contour
    paths (`paths`) to the given `save_path` for the given rater, subject ID
    (`sid`) and hemisphere (`h`). The return value is a dict whose keys are the
    path names and whose values are the filenames to which the path was saved.

    Parameters
    ----------
    rater : str or None
        The name of the rater into whose directory these tracess should be
        saved. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being saved. If
        `None` is given, then the subject id is not included in the output
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        saved.
    paths : dict
        A dictionary whose keys are the names of the paths to be saved and
        whose values are the paths themselves (which must be neuropythy
        `Path` objects).
    save_path : path-like or None
        The directory into which the annotations are to be saved. Note that this
        directory is the base directory that is combined with the rater and
        sid to produce an actual save directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the paths that are to be
        saved and whose values are the filename templates. If this option is
        excluded or provided the value `None`, then all of the paths in
        `paths` are saved and the filename template is
        `'{hemisphere}.{name}_path.json.gz'`. If `filenames` is a string, then
        it is looked up in the `'paths'` step of the processing data using the
        `procdata` function (e.g. if `filenames == 'ventral'` then the dict
        `procdata('ventral', 'paths')` is used. If the filenames dict contains
        an entry whose key is `None`, then that entry is used as a catch-all for
        paths whose names are not explicitly listed. The filename template(s)
        may use the `{hemisphere}`, `{sid}`, `{rater}`, and `{name}` format ids.
    overwrite : boolean or None, optional
        Whether to overwrite files if they already exist. If `False` is given,
        then existing files are not ovewritten and an exception is raised if an
        existing file is found. If `True` is given (the default) then existing
        files are overwritten. If `None` is given, then existing contour files
        are not overwritten, but no errors are raised if they are found.
    mkdir : boolean, optional
        Whether to create the path directory if it does not exist. If `True` is
        given (the default), then all parent directories and the requested
        directory are created with the mode given by `mkdir_mode`. If `False` is
        given, then no directories are created.
    mkdir_mode : int, optional
        The mode to be used when creating directories. The default is `0o775`.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` parameter and
        whose values are the paths of the files that were successfully saved.
    """
    return _save_annotdata(
        'paths', _save_path,
        rater, sid, h, paths,
        save_path=save_path,
        filenames=filenames,
        default_filename='{hemisphere}.{name}_path.json.gz',
        overwrite=overwrite,
        mkdir=mkdir,
        mkdir_mode=mkdir_mode)
def load_paths(rater, sid, h, filenames,
               load_path=None,
               cortex=None,
               missing_okay=False):
    """Loads a set of paths for a subject and hemisphere from a save path.

    `load_paths(rater, sid, h, filenames, load_path)` loads a set of contours
    from the given `load_path` for the given rater, subject ID (`sid`) and
    hemisphere (`h`). These paths are returned as a dict whose keys are the
    paths names and whose values are the neuropythy `Path` objects. The path
    names and filenames are provided in the `filenames` argument which must be a
    dictionary whose keys are trace names and whose values are filename
    templates (see below).

    Parameters
    ----------
    rater : str or None
        The name of the rater from whose directory these annotationss should be
        loaded. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being loaded. If
        `None` is given, then the subject id is not included in the input
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        loaded.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the annotationss that are to
        be loaded and whose values are the filename templates. If `filenames` is
        a string, then it is looked up in the `'paths'` step of the
        processing data using the `procdata` function (e.g. if `filenames ==
        'ventral'` then the dict `procdata('ventral', 'paths')` is used.
    load_path : path-like or None
        The directory from which the annotationss are to be loaded. Note that
        this directory is the base directory that is combined with the rater and
        sid to produce an actual load directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    cortex : neuropythy Cortex or None, optional
        The cortex object to attach to use to create the path object. If `None`
        is given, then `cortex(sid, h)` function from `hcpannot.config` is used.
    missing_okay : boolean, optional
        Whether it is okay for some or all of the files in `filenames` to be missing.
        If `True` then errors are not raised when a missing file is encountered and
        instead no entry in the resulting dictionary is included. If `False` (the
        default), then an error is raised if a file is missing.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` argument and
        whose values are the neuropythy `Path` objects that were successfully
        loaded.
    """
    if cortex is None:
        from .config import cortex
        loadfn = lambda filepath: _load_path(filepath, cortex(sid, h))
    else:
        loadfn = lambda filepath: _load_path(filepath, cortex)
    return _load_annotdata(
        'paths', loadfn,
        rater, sid, h, filenames,
        load_path=load_path,
        missing_okay=missing_okay)


# Labels #######################################################################

def _save_label(filepath, label):
    "Save a single label-vector to a filename"
    ny.save(os.fspath(filepath), label, 'mgh')
    return filepath
def _load_label(filepath):
    "Loads a single label-vector from the given filepath."
    return ny.load(os.fspath(filepath), 'mgh', to='field')
def save_labels(rater, sid, h, labels,
                save_path=None, filenames=None,
                overwrite=True, mkdir=True, mkdir_mode=0o775):
    """Saves a set of labels for a subject and hemisphere to a save path.

    `save_labels(rater, sid, h, contours, save_path)` saves a set of labels
    (`labels`) to the given `save_path` for the given rater, subject ID (`sid`)
    and hemisphere (`h`). The return value is a dict whose keys are the label
    names and whose values are the filenames to which the label was saved.

    Parameters
    ----------
    rater : str or None
        The name of the rater into whose directory the annotations should be
        saved. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being saved. If
        `None` is given, then the subject id is not included in the output
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        saved.
    labels : dict
        A dictionary whose keys are the names of the labels to be saved and
        whose values are the labels themselves (which must be vectors).
    save_path : path-like or None
        The directory into which the annotations are to be saved. Note that this
        directory is the base directory that is combined with the rater and
        sid to produce an actual save directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the labels that are to be
        saved and whose values are the filename templates. If this option is
        excluded or provided the value `None`, then all of the labels in
        `labels` are saved and the filename template is
        `'{hemisphere}.{name}.mgz'`. If `filenames` is a string, then
        it is looked up in the `'labels'` step of the processing data using the
        `procdata` function (e.g. if `filenames == 'ventral'` then the dict
        `procdata('ventral', 'labels')` is used. If the filenames dict contains
        an entry whose key is `None`, then that entry is used as a catch-all for
        labels whose names are not explicitly listed. The filename template(s)
        may use the `{hemisphere}`, `{sid}`, `{rater}`, and `{name}` format ids.
    overwrite : boolean or None, optional
        Whether to overwrite files if they already exist. If `False` is given,
        then existing files are not ovewritten and an exception is raised if an
        existing file is found. If `True` is given (the default) then existing
        files are overwritten. If `None` is given, then existing contour files
        are not overwritten, but no errors are raised if they are found.
    mkdir : boolean, optional
        Whether to create the path directory if it does not exist. If `True` is
        given (the default), then all parent directories and the requested
        directory are created with the mode given by `mkdir_mode`. If `False` is
        given, then no directories are created.
    mkdir_mode : int, optional
        The mode to be used when creating directories. The default is `0o775`.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` parameter and
        whose values are the paths of the files that were successfully saved.

    """
    return _save_annotdata(
        'labels', _save_label,
        rater, sid, h, labels,
        save_path=save_path,
        filenames=filenames,
        default_filename='{hemisphere}.{name}.mgz',
        overwrite=overwrite,
        mkdir=mkdir,
        mkdir_mode=mkdir_mode)
def load_labels(rater, sid, h, filenames,
                load_path=None,
                missing_okay=False):
    """Loads a set of labels for a subject and hemisphere from a save path.

    `load_labels(rater, sid, h, filenames, load_path)` loads a set of contours
    from the given `load_path` for the given rater, subject ID (`sid`) and
    hemisphere (`h`). These labels are returned as a dict whose keys are the
    labels names and whose values are the label vectors. The label names and
    filenames are provided in the `filenames` argument which must be a
    dictionary whose keys are label names and whose values are filename
    templates (see below).

    Parameters
    ----------
    rater : str or None
        The name of the rater from whose directory these annotationss should be
        loaded. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being loaded. If
        `None` is given, then the subject id is not included in the input
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        loaded.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the annotationss that are to
        be loaded and whose values are the filename templates. If `filenames` is
        a string, then it is looked up in the `'labels'` step of the
        processing data using the `procdata` function (e.g. if `filenames ==
        'ventral'` then the dict `procdata('ventral', 'labels')` is used.
    load_path : path-like or None
        The directory from which the annotationss are to be loaded. Note that
        this directory is the base directory that is combined with the rater and
        sid to produce an actual load directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    missing_okay : boolean, optional
        Whether it is okay for some or all of the files in `filenames` to be
        missing.  If `True` then errors are not raised when a missing file is
        encountered and instead no entry in the resulting dictionary is
        included. If `False` (the default), then an error is raised if a file is
        missing.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` argument and
        whose values are the label vectors that were successfully loaded.
    """
    return _load_annotdata(
        'labels', _load_label,
        rater, sid, h, filenames,
        load_path=load_path,
        missing_okay=missing_okay)


# Reports ######################################################################

def _save_report(filepath, report):
    "Save a single report dictionary to a filename"
    ny.save(os.fspath(filepath), report, 'json')
    return filepath
def _load_report(filepath):
    "Loads a single report dictionary from the given filepath."
    return ny.load(os.fspath(filepath), 'json')
def save_reports(rater, sid, h, reports,
                 save_path=None, filenames=None,
                 overwrite=True, mkdir=True, mkdir_mode=0o775):
    """Saves a set of reports for a subject and hemisphere to a save path.

    `save_reports(rater, sid, h, contours, save_path)` saves a set of reports
    (`reports`) to the given `save_path` for the given rater, subject ID (`sid`)
    and hemisphere (`h`). The return value is a dict whose keys are the report
    names and whose values are the filenames to which the report was saved.

    Parameters
    ----------
    rater : str or None
        The name of the rater into whose directory the annotations should be
        saved. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being saved. If
        `None` is given, then the subject id is not included in the output
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        saved.
    reports : dict
        A dictionary whose keys are the names of the reports to be saved and
        whose values are the reports themselves (which must be JSON-serializable
        objects).
    save_path : path-like or None
        The directory into which the annotations are to be saved. Note that this
        directory is the base directory that is combined with the rater and
        sid to produce an actual save directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the reports that are to be
        saved and whose values are the filename templates. If this option is
        excluded or provided the value `None`, then all of the reports in
        `reports` are saved and the filename template is
        `'{hemisphere}.{name}.mgz'`. If `filenames` is a string, then
        it is looked up in the `'reports'` step of the processing data using the
        `procdata` function (e.g. if `filenames == 'ventral'` then the dict
        `procdata('ventral', 'reports')` is used. If the filenames dict contains
        an entry whose key is `None`, then that entry is used as a catch-all for
        reports whose names are not explicitly listed. The filename template(s)
        may use the `{hemisphere}`, `{sid}`, `{rater}`, and `{name}` format ids.
    overwrite : boolean or None, optional
        Whether to overwrite files if they already exist. If `False` is given,
        then existing files are not ovewritten and an exception is raised if an
        existing file is found. If `True` is given (the default) then existing
        files are overwritten. If `None` is given, then existing contour files
        are not overwritten, but no errors are raised if they are found.
    mkdir : boolean, optional
        Whether to create the path directory if it does not exist. If `True` is
        given (the default), then all parent directories and the requested
        directory are created with the mode given by `mkdir_mode`. If `False` is
        given, then no directories are created.
    mkdir_mode : int, optional
        The mode to be used when creating directories. The default is `0o775`.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` parameter and
        whose values are the paths of the files that were successfully saved.

    """
    return _save_annotdata(
        'reports', _save_report,
        rater, sid, h, reports,
        save_path=save_path,
        filenames=filenames,
        default_filename='{hemisphere}.{name}_report.json',
        overwrite=overwrite,
        mkdir=mkdir,
        mkdir_mode=mkdir_mode)
def load_reports(rater, sid, h, filenames,
                 load_path=None,
                 missing_okay=False):
    """Loads a set of reports for a subject and hemisphere from a save path.

    `load_reports(rater, sid, h, filenames, load_path)` loads a set of contours
    from the given `load_path` for the given rater, subject ID (`sid`) and
    hemisphere (`h`). These reports are returned as a dict whose keys are the
    reports names and whose values are the report objects. The report names and
    filenames are provided in the `filenames` argument which must be a
    dictionary whose keys are report names and whose values are filename
    templates (see below).

    Reports can be any JSON-serializable object.

    Parameters
    ----------
    rater : str or None
        The name of the rater from whose directory these annotationss should be
        loaded. If `None` is given, then a directory for the rater is not
        included.
    sid : str or int or None
        The string or integer id of the subject whose data is being loaded. If
        `None` is given, then the subject id is not included in the input
        directory.
    h : str
        The hemisphere name (`'lh'` or `'rh'`) of the annotations that are being
        loaded.
    filenames : dict or None, optional
        The dictionary whose keys are the names of the annotationss that are to
        be loaded and whose values are the filename templates. If `filenames` is
        a string, then it is looked up in the `'reports'` step of the
        processing data using the `procdata` function (e.g. if `filenames ==
        'ventral'` then the dict `procdata('ventral', 'reports')` is used.
    load_path : path-like or None
        The directory from which the annotationss are to be loaded. Note that
        this directory is the base directory that is combined with the rater and
        sid to produce an actual load directory. If `None`, then the current
        working directory is used. See also `to_data_path()`.
    missing_okay : boolean, optional
        Whether it is okay for some or all of the files in `filenames` to be missing.
        If `True` then errors are not raised when a missing file is encountered and
        instead no entry in the resulting dictionary is included. If `False` (the
        default), then an error is raised if a file is missing.

    Returns
    -------
    dict
        A dictionary whose keys are the keys in the `filenames` argument and
        whose values are the report objects that were successfully loaded.
    """
    return _load_annotdata(
        'reports', _load_report,
        rater, sid, h, filenames,
        load_path=load_path,
        missing_okay=missing_okay)
