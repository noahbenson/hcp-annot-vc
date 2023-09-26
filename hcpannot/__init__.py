'''
The hcpannot package contains code for annotating the HCP 7 Tesla Retinotopy
Dataset (DOI: 10.1167/18.13.23).

These tools are intended to be run using a Docker container; for information
on how to use these tools, see the README.md file in the github repository
noahbenson/hcp-annot-vc.
'''

from .config import (
    subject_list,
    procdata)
from .core import (
    plot_angle, plot_eccen, plot_curv, plot_hmbound,
    plot_vmbound, plot_as_image, op_flatmap, plot_isoang,
    plot_isoecc, generate_images, label_to_segs)
from .interface import (
    subject_data, ROITool,
    imgrid_to_flatmap, flatmap_to_imgrid)
from .analysis import (plot_contours)
from .io import (
    load_contours, load_traces, load_paths, load_labels, load_reports,
    save_contours, save_traces, save_paths, save_labels, save_reports)

from . import config
