#! /usr/bin/env python
################################################################################

import os
import pandas as pd

import hcpannot
import hcpannot.cmd as hcpa_cmd

from hcpannot.mp import (makejobs, mprun)
from hcpannot.proc import meanproc_all

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_means.py',
    description='Processes the mean contours one hemisphere at a time.')
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
if overwrite is False:
    overwrite = None
nproc = hcpa_conf.opts['nproc']


# Running the Jobs #############################################################

# Make the job list.
opts = dict(
    save_path=save_path,
    load_path=load_path,
    overwrite=overwrite,
    source_raters=raters)
def call_proc_all(sid, h):
    return meanproc_all('ventral', sid=sid, hemisphere=h, **opts)
def firstarg(a, b):
    return a
jobs = makejobs(sids, hemis)
# Run this step in the processing.
dfs = proc_traces_results = mprun(
    call_proc_all, jobs, "meanventral",
    nproc=nproc,
    onfail=firstarg,
    onokay=firstarg)
df = pd.concat(dfs)
df.to_csv(
    os.path.join(save_path, 'proc_meanventral.tsv'),
    sep='\t',
    index=False)
