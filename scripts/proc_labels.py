#! /usr/bin/env python
################################################################################

import hcpannot
import hcpannot.cmd as hcpa_cmd

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_labels.py',
    description='Parses contour trace files into labels.')
hcpa_conf.parser.add_argument(
    '-w', '--no-output-weights',
    default=True,
    action='store_const',
    const=False,
    dest='output_weights',
    help='Whether to output weight files for the labels.')
hcpa_conf.parser.add_argument(
    '-v', '--no-output-volume',
    default=True,
    action='store_const',
    const=False,
    dest='output_volume',
    help='Whether to output a volume file for the labels.')
args = hcpa_conf.parsed_args()
hcpa_conf.opts['output_weights'] = args.output_weights
hcpa_conf.opts['output_volume'] = args.output_volume

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
from hcpannot.io import export_labels

# Make the job list.
jobs = makejobs([raters], sids, [opts])
# Run this step in the processing.
proc_labels_results = mpstep(
    export_labels, jobs, "labels", save_path,
    overwrite=overwrite,
    nproc=nproc)
