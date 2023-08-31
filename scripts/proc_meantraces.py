#! /usr/bin/env python
################################################################################

import hcpannot
import hcpannot.cmd as hcpa_cmd

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_meantraces.py',
    description='Processes the mean contours into trace files.')
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

# There are three jobs, really; one that calculates the means, one that then
# processes the traces for the means, and finally one that processes the paths.
from hcpannot.mp       import (makejobs, mpstep)
from hcpannot.io       import (export_means, export_traces, export_paths)
from hcpannot.analysis import meanrater

opts = dict(
    save_path=save_path, load_path=load_path, overwrite=overwrite,
    vc_contours='ventral_meanrater')
jobs = makejobs([meanrater], sids, hemis, [opts])
jobs = sorted(jobs, key=lambda job: (job[1], job[2]))
proc_meantraces_results = mpstep(
    export_traces, jobs, "meantraces", save_path,
    overwrite=overwrite,
    nproc=nproc)
