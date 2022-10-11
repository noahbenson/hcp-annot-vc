'''
The hcpannot package contains code for annotating the HCP 7 Tesla Retinotopy
Dataset (DOI: 10.1167/18.13.23).

These tools are intended to be run using a Docker container; for information
on how to use these tools, see the README.md file in the github repository
noahbenson/hcp-annot-vc.
'''

from .core import (plot_angle, plot_eccen, plot_curv, plot_hmbound,
                   plot_vmbound, plot_as_image, op_flatmap, plot_isoang,
                   plot_isoecc, generate_images, label_to_segs)
from .interface import (subject_ids, subject_data, ROITool,
                        imgrid_to_flatmap, flatmap_to_imgrid)
from .analysis import (vc_plan, to_data_path, save_contours, load_contours,
                       subject_list_1, subject_list_2, subject_list_3)
from .io import (load_traces, load_paths)
