#! /usr/bin/env python
################################################################################

import hcpannot
import hcpannot.cmd as hcpa_cmd

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_paths.py',
    description='Parses trace files into path files for each subject.')
hcpannot.interface.default_load_path = hcpa_conf.opts['cache_path']

raters = hcpa_conf.raters
if raters is None:
    raters = hcpa_cmd.default_raters['ventral']
sids = hcpa_conf.sids
hemis = hcpa_conf.hemis
opts = hcpa_conf.opts
save_path = hcpa_conf.opts['save_path']
load_path = hcpa_conf.opts['load_path']
overwrite = hcpa_conf.opts['overwrite']
nproc = hcpa_conf.opts['nproc']

# For calculating surface areas, we use okay and fail functions that return an
# okaystr or failstr bur that also records a dict of data for the dataframe.
def sarea_fail(err, tup):
    import numpy as np
    from hcpannot.mp import failstr
    msg = failstr(err, tup)
    (rater, sid, h, opts) = tup
    res = dict(
        rater=rater, sid=sid, hemisphere=h,
        hV4=np.nan, VO1=np.nan, VO2=np.nan, cortex=np.nan,
        message=msg)
    return res
def sarea_okay(res, tup):
    from hcpannot.mp import okaystr
    msg = okaystr(res, tup)
    res = dict(res, message=msg)
    return res


# Running the Jobs #############################################################

# There are three jobs, really; one that calculates the means, one that then
# processes the traces for the means, and finally one that processes the paths.
from hcpannot.mp       import (makejobs, mprun)
from hcpannot.io       import (calc_surface_areas)
from hcpannot.analysis import (meanrater)
import neuropythy as ny

opts = dict(load_path=load_path)
jobs = makejobs(raters + [meanrater], sids, hemis, [opts])
if overwrite or not os.path.isfile(save_path):
    proc_sarea_results = mprun(
        calc_surface_areas, jobs, "sarea",
        nproc=nproc,
        onokay=sarea_okay,
        onfail=sarea_fail)
    # Convert the surface area data to a dataframe and save it.
    sarea_data = ny.to_dataframe(proc_sarea_results)
    ny.save(save_path, sarea_data)
