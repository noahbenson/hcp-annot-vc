################################################################################
# mp.py
#
# A set of functions for managing multiprocessing jobs.
# by Noah C. Benson <nben@uw.edu>

# Import things
import sys, os, pimms, json, urllib

import numpy as np
import pyrsistent as pyr
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import neuropythy as ny


# Multiprocessing ##############################################################
# This function gets used below to pass tuples of arguments to functions through
# the multiprocessing pool interface.
def tupcall(fn, tup=None,
            onfail=None,
            onokay=None,
            retryon=None):
    """Passes a tuple of arguments to a function.

    `tupcall(fn, tuple)` calls `fn(*tuple)` and returns the result. If the final
    element of `tuple` is a dictionary, then the return value is instead
    `fn(*tuple[:-1], **tuple[-1])`. An empty dictionary may be passed as the
    final element of `tuple`.
    
    `tupcall(fn)` returns a function `g` such that `g(tuple)` is equivalent to
    `tupcall(fn, tuple)`.
    
    The optional argument `onfail` can be set to a backup function that is 
    called if an exception is raised. In such a case, the return value of the
    `tupcall` function is `onfail(raised_exception, tuple)`.
    
    The optional argument `onokay` can be set to a function that is 
    called if no exception is raised. In such a case, the return value of the
    `tupcall` function is `onokay(raised_error, tuple)`.

    The optional argument `retryon` can be set to an exception type or a tuple
    of exception types; if one of these exceptions is raised during execution,
    then the function is retried once. Alternately, `retryon` may be a dict
    whose keys are exception types or tuples of exception types and whose values
    are functions that determine whether or not to retry the call. These
    functions are called as `do_retry = fn(raised_error, tuple)`, and `do_retry`
    must be a boolean result.
    """
    if tup is None:
        from functools import partial
        return partial(tupcall, fn,
                       onfail=onfail, onokay=onokay, retryon=retryon)
    elif not isinstance(tup, (tuple, list)):
        raise ValueError("tupcall requires a tuple or list")
    elif len(tup) == 0:
        fin = {}
        arg = tup
    else:
        fin = tup[-1]
        if isinstance(fin, dict):
            arg = tup[:-1]
        else:
            fin = {}
            arg = tup
    if retryon is None and onfail is None:
        res = fn(*arg, **fin)
    else:
        while True:
            try:
                res = fn(*arg, **fin)
                break
            except Exception as e:
                if retryon is not None:
                    if isinstance(retryon, dict):
                        do_retry = False
                        for (k,fn) in retryon.items():
                            if isinstance(e, k):
                                do_retry = fn(e, tup)
                                if do_retry: break
                        if do_retry: continue
                    elif isinstance(e, retryon):
                        # We only retry once.
                        retryon = None
                        continue
                # If we reach here, we aren't retrying.
                if onfail:
                    return onfail(e, tup)
                else:
                    raise                    
    if onokay is not None:
        return onokay(res, tup)
    else:
        return res
def okaystr(res, tup):
    """Returns a string summary of an okay result for use with `tupcall`.
    
    `okaystr(r, tuple)` returns a string `f"OKAY: {tuple}"`.
    """
    return f"OKAY: {tup}"
def failstr(err, tup):
    """Returns a string summary of a failed result for use with `tupcall`.
    
    `failstr(err, tuple)` returns a string `f"FAIL: {tuple} {err}"`.
    """
    return f"FAIL: {tup} {err}"
def retry_sleep(err=None, tup=None, duration=Ellipsis):
    """Sleeps for 5 seconds then returns `True`, for use with `tupcall`.

    `retry_sleep(error, tuple)` sleeps for 5 seconds then returns `True`.
    
    `retry_sleep(error, tuple, duration=dur)` sleeps for `dur` seconds then
    returns `True`.
    
    `retry_sleep(dur)` returns a function equivalent to
    `lambda err, tup: retry_sleep(err, tup, duration=dur)`.
    
    This function is intended for use with `tupcall`'s `retryon` option, for
    example, `tupcall(fn, tup, retryon={HTTPError: retry_sleep(5)})`.
    """
    import time
    if err is None and tup is None:
        dur = duration
    elif tup is None and duration is Ellipsis:
        dur = err
    elif tup is None:
        raise ValueError(f"invalid arguments to retry_sleep:"
                         f" ({err},{tup},{duration})")
    else:
        dur = 5 if duration is Ellipsis else duration
        time.sleep(dur)
        return True
    # If we make it here, we need to return a partial function.
    return lambda err,tup: retry_sleep(err, tup, duration=dur)
# We actually want to retry on HTTPError, which occurs when we overload OSF with
# download requests.
def mprun(fn, jobs,
          tag=None,
          nproc=None,
          print=print,
          onokay=okaystr,
          onfail=failstr,
          retryon={urllib.error.HTTPError: retry_sleep}):
    """Runs a function across many processes via the `multiprocessing` package.
    
    `mprun(fn, joblist)` runs the given function across as many processes as
    there are CPUs for each job in `joblist`. The jobs should be tuples that
    can be executed via `tupcall(fn, job)`.
    
    The optional arguments `onfail` and `onokay` are passed through to the
    `tupcall` function. Additionally, the option `nproc` can be set to the
    number of processes that should be used; the default of `None` indicates
    that the number of processes should match the number of CPUs. Finally, the
    option `print` may be set to a print function that is used to log the
    progress of the run. If `None` is given, then no printing is done;
    otherwise, every 10% of the total set of jobs complete produces a progress
    message.
    """
    import multiprocessing as mp
    # Process the arguments.
    njobs = len(jobs)
    try:
        fnname = fn.__name__
    except Exception:
        fnname = str(fn)
    if nproc is None:
        nproc = mp.cpu_count()
    callfn = tupcall(fn, onfail=onfail, onokay=onokay, retryon=retryon)
    # Start the jobs!
    print(f"Beginning {njobs} jobs with tag '{tag}'...")
    donecount = 0
    res = []
    for ii in range(0, njobs, nproc):
        jj = min(ii + nproc, njobs)
        nn = jj - ii
        with mp.Pool(nn) as pool:
            r = pool.map(callfn, jobs[ii:jj])
        for rr in r:
            res.append(rr)
        if donecount * 10 // njobs < (donecount + nn) * 10 // njobs:
            print(" - %4d / %4d (%3d%%)" % (jj, njobs, int(100*jj/njobs)))
        donecount += nn
    return res
def mpstep(fn, jobs, tag, save_path,
           overwrite=False, nproc=None,
           onokay=okaystr, onfail=failstr, print=print,
           retryon={urllib.error.HTTPError: retry_sleep}):
    """Runs one multiprocessing step in the contour process/export workflow.
    
    `mpstep(fn, jobs, tag, save_path)` is designed to be run with the
    `hcpannot.io` functions for exporting processed data about the contours:
      * `export_traces`
      * `export_paths`
      * `export_means`
      * `export_labels`
    Each of these functions must be multiprocessed across many combinations of
    raters, subjects, and hemispheres. These arguments must be listed in `jobs`.
    The `tag` is used to name the logfile that is exported on success.
    
    Parameters
    ----------
    fn : function
        The function that is to be multiprocessed across all jobs.
    jobs : list of tuples
        A list of arguments to `fn`; see `makejobs` and `tupcall`.
    tag : str
        A tag name used to identify the logfile, which is placed in the
        `save_path` directory.
    save_path : directory name
        The directory from which to load the contour data and into which to
        write the logfile.
    overwrite : boolean, optional
        If overwrite is `False` and the logfile already exists, then it is read
        in and returned instead of rerunning the jobs. The default is `False`.
    nproc : int or None, optional
        The number of processes to multiplex across. If this is `None`, then the
        number of CPUs is used. The default is `None`.
    
    Returns
    -------
    list of str
        A list with one entry per job that is the `okaystr` or `failstr`
        representation of that job's return value.
    """
    # We'll write out this logfile.
    #logfile = f"{tag}_{datetime.datetime.now().isoformat()}.log"
    logfile = f"proc_{tag}.log"
    logfile = os.path.join(save_path, logfile)
    # Multiprocess all the jobs.
    if overwrite or not os.path.isfile(logfile):
        proc_results = mprun(fn, jobs,
                             tag=tag,
                             nproc=nproc,
                             print=print,
                             onokay=onokay,
                             onfail=onfail,
                             retryon=retryon)
        # Write out a log of these results.
        with open(logfile, "wt") as fl:
            fl.write('\n'.join(proc_results))
            fl.write('\n')
    else:
        with open(logfile, "rt") as fl:
            proc_results = fl.readlines()
    return proc_results
def makejobs(*args):
    "Equivalent to `itertools.product(*args)` but always returns a list."
    from itertools import product
    return list(product(*args))
