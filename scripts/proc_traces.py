#! /usr/bin/env python
################################################################################

import hcpannot
import hcpannot.cmd as hcpa_cmd

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_traces.py',
    description='Parses contour files into trace files.')
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


# Running the Jobs #############################################################
from hcpannot.mp import (makejobs, mpstep)
from hcpannot.io import export_traces

# Make the job list.
opts = dict(save_path=save_path, load_path=load_path, overwrite=overwrite)
jobs = makejobs(raters, sids, hemis, [opts])
# Run this step in the processing.
proc_traces_results = mpstep(
    export_traces, jobs, "traces", save_path,
    overwrite=overwrite,
    nproc=nproc)
