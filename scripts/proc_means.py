#! /usr/bin/env python
################################################################################

import os
import pandas as pd

import hcpannot
import hcpannot.cmd as hcpa_cmd

from hcpannot.mp import (makejobs, mprun)
from hcpannot.proc import (allproc_meanrater, allproc_meansub)

def firstarg(arg0, *args):
    return arg0

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_means.py',
    description='Processes the mean contours one hemisphere at a time.')
hcpannot.interface.default_load_path = hcpa_conf.opts['cache_path']

region = hcpa_conf.region
raters = hcpa_conf.raters
if raters is None:
    raters = hcpa_cmd.default_raters[region]
sids = hcpa_conf.sids
hemis = hcpa_conf.hemis
opts = hcpa_conf.opts
save_path = hcpa_conf.opts['save_path']
load_path = hcpa_conf.opts['load_path']
overwrite = hcpa_conf.opts['overwrite']
if overwrite is False:
    overwrite = None
nproc = hcpa_conf.opts['nproc']
if region not in ('ventral', 'dorsal'):
    raise ValueError(f"region must be ventral or dorsal; got {region}")


# Running the Jobs #############################################################

# First, we do the mean rater --------------------------------------------------
# Make the job list.
opts = dict(
    save_path=save_path,
    load_path=load_path,
    overwrite=overwrite,
    source_raters=raters)
def call_allproc_meanrater(sid, h):
    return allproc_meanrater(region, sid=sid, hemisphere=h, **opts)
jobs = makejobs(sids, hemis)
# Run this step in the processing.
dfs_rater = mprun(
    call_allproc_meanrater, jobs, f"meanrater_{region}",
    nproc=nproc,
    onfail=firstarg,
    onokay=firstarg)
df_rater = pd.concat(dfs_rater)

# Next, do the mean subject ----------------------------------------------------
opts = dict(
    save_path=save_path,
    load_path=load_path,
    overwrite=overwrite,
    source_sids=sids)
def call_allproc_meansub(rater, h):
    return allproc_meansub(region, rater=rater, hemisphere=h, **opts)
jobs = makejobs(raters + ['mean'], hemis)
# Run this step in the processing.
dfs_sub = mprun(
    call_allproc_meansub, jobs, f"meansub_{region}",
    nproc=nproc,
    onfail=firstarg,
    onokay=firstarg)
df_sub = pd.concat(dfs_sub)

df = pd.concat([df_rater, df_sub])
df.to_csv(
    os.path.join(save_path, f'proc_mean{region}.tsv'),
    sep='\t',
    index=False)
