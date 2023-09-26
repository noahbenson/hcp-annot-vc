################################################################################
# proc/ventral.py
#
# Pipeline for processing subjects and saving/loading the outputs to disk.

"""The ventral contours processing workflow.

This file contains the processing workflow for the ventral cortical contours, as
drawn for the HCP visual cortex annotation project. The code in this file
converts a set of contours 

"""


# Dependencies #################################################################

import sys, os, pimms, json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import neuropythy as ny

from ..config import (
    meanrater,
    procdata)
from ..io import (
    save_contours, load_contours,
    save_traces,   load_traces,
    save_paths,    load_paths,
    save_labels,   load_labels,
    save_reports,  load_reports)
from .core import (
    init_plan,
    init_meanplan)


# Ventral Processing ###########################################################

# The calculations -------------------------------------------------------------
@pimms.calc('v3v_contour')
def load_v3v_contour(sid, chirality):
    """Loads the V3-ventral contour from the HCP-lines dataset.
    
    Parameters
    ----------
    sid : int
        The HCP subject-ID of the subject to load.
    chirality : 'lh' or 'rh'
        The hemisphere (`'lh'` or `'rh'`) to load.
        
    Outputs
    -------
    v3v_contour : NumPy array
        A NumPy array of the points in the V3-ventral contour.
    """
    from ..interface import subject_data
    sdat = subject_data[(sid, chirality)]
    v3v = sdat['v123']['V3_ventral']
    v3v = np.array(v3v)
    # This is the v3-ventral contour.
    return (v3v,)
def _extend_contour(pts, d=100):
    (u, v) = (pts[:,0] - pts[:,1], pts[:,-1] - pts[:,-2])
    u /= np.sqrt(np.sum(u**2))
    v /= np.sqrt(np.sum(v**2))
    pts_ext = [(u*d + pts[:,0])[:,None], pts, (v*d + pts[:,-1])[:,None]]
    pts_ext = np.hstack(pts_ext)
    return pts_ext
@pimms.calc('preproc_contours', 'ext_contours', 'outer_sources')
def calc_extended_contours(rater, contours, chirality, v3v_contour):
    """Creates and extended hV4/VO1 boundary contour.
    
    The contours are extended by adding points to either end that are a 
    distance of 100 map units from the respective endpoints at an angle of 180
    degrees to the immediate-next interior point.
    
    Parameters
    ----------
    contours : dict-like
        The dictionary of contours.
    v3v_contour : NumPy array
        The `(2 x N)` matrix of points in the V3-ventral contour.
        
    Outputs
    -------
    preproc_contour : dict
        A dictionary whose keys are the same as in `contours` but that have
        been mildly preprocessed: all contours are ordered starting from the
        end nearest the V3-ventral boundary. Additionally the `'outer'` contour
        has been added, consisting of the hV4-outer, V3-ventral, and VO-outer
        contours conjoined, starting at the hV4-outer and ending with the
        VO-outer.
    ext_contours : dict
        A dictionary whose keys are the same as in `preproc_contour` but whose
        ends have been extended by 100 map units in the same directoin as the
        ending segments.
    outer_sources : array of str
        A list whose elements give the origin of each point in the `'outer'`
        contour. Each element will be either `'hV4_outer'`, `'VO_outer'`, or
        `'V3v'`.
    """
    contours = dict(contours)
    if any(k not in contours for k in ('outer', 'hV4_VO1', 'VO1_VO2')):
        contours['V3_ventral'] = v3v_contour
        # Both the hV4-VO1 contour need to be re-ordered to be starting near
        # the V3-ventral contour and ending far from it.
        v3v_end = v3v_contour[:,0]
        for c in ('hV4_VO1', 'VO1_VO2'):
            pts = contours[c]
            ends = pts[:,[0,-1]].T
            dists = [np.sqrt(np.sum((end - v3v_end)**2)) for end in ends]
            if dists[1] < dists[0]:
                pts = np.fliplr(pts)
                contours[c] = pts
        # Make the outer contour:
        outer = [np.fliplr(contours['hV4_outer']),
                 np.fliplr(v3v_contour),
                 contours['VO_outer']]
        outer_sources = (['hV4_outer'] * (outer[0].shape[1] + 1) + 
                         ['V3v'] * outer[1].shape[1] +
                         ['VO_outer'] * (outer[2].shape[1] + 1))
        outer = np.hstack(outer)
        outer_sources = np.array(outer_sources)
        contours['outer'] = outer
    else:
        outer_sources = None
    # Now make the extended contours:
    ext_contours = {k:_extend_contour(c) for (k,c) in contours.items()}
    # And return!
    return (contours, ext_contours, outer_sources)
def _cross_isect(segs1, segs2, rtol=1e-05, atol=1e-08):
    from neuropythy.geometry.util import segment_intersection_2D
    (segs1, segs2) = (np.asarray(segs1), np.asarray(segs2))
    (n1, n2) = (segs1.shape[1] - 1, segs2.shape[1] - 1)
    ii1 = np.concatenate([[k]*n2 for k in range(n1)])
    ii2 = np.arange(n2)
    ii2 = np.concatenate([ii2]*n1)
    pts = segment_intersection_2D([segs1[...,ii1], segs1[...,ii1+1]],
                                  [segs2[...,ii2], segs2[...,ii2+1]],
                                  inclusive=True)
    pts = np.asarray(pts)
    ii = np.all(np.isfinite(pts), axis=0)
    (ii1, ii2, pts) = (ii1[ii], ii2[ii], pts[:,ii])
    # We have to be careful with sorting: if two of the ii1 segments are
    # the same, we want to order them by the distance from the segment
    # start.
    seglen2s = np.sum((segs1[:, :-1] - segs1[:, 1:])**2, axis=0)
    ds = [
        np.sum(seglen2s[:ii1[ii]]) + np.sum((segs1[:, ii1[ii]] - pts[:,ii])**2)
        for ii in range(len(ii1))]
    ii = np.argsort(ds)
    (ii1, ii2, pts) = (ii1[ii], ii2[ii], pts[:,ii])
    # See if we have accidentally created identical points.
    for ii in reversed(range(pts.shape[1] - 1)):
        if np.isclose(pts[:,ii], pts[:,ii+1], atol=atol, rtol=rtol).all():
            ii1 = np.delete(ii1, ii+1)
            ii2 = np.delete(ii2, ii+1)
            pts = np.delete(pts, ii+1, axis=-1)
    return (ii1, ii2, pts)
def _closer(x, a, b):
    da2 = np.sum((x - a)**2)
    db2 = np.sum((x - b)**2)
    return (da2 < db2)
@pimms.calc('normalized_contours', 'boundaries')
def calc_normalized_contours(sid, hemisphere, rater, outer_sources,
                             preproc_contours, ext_contours):
    """Normalizes the raw contours and converts them into path-traces.
    """
    hv4_vo1 = ext_contours['hV4_VO1']
    vo1_vo2 = ext_contours['VO1_VO2']
    outer = ext_contours['outer']
    # We want to find the intersections between hV4-VO1 and the outer boundary.
    # There must be 3; any other number is an error.
    (hii, oii, pts_ho) = _cross_isect(hv4_vo1, outer)
    if len(hii) > 3:
        # We need to figure out which three intersections of the hV4-VO1 border
        # and the outer border we're going to use.
        # One of these intersections must be near the V3v border, so we use the
        # outer sources to figure that out.
        if outer_sources is None:
            raise RuntimeError(f"{len(hii)} hV4-VO1 / Outer intersections for "
                               f"{rater}/{sid}/{hemisphere}, which has no "
                               f"outer_sources data")
        v3v_start = np.where(outer_sources == 'V3v')[0][0]
        voo_start = np.where(outer_sources == 'VO_outer')[0][0]
        v3v_dist = np.abs(voo_start - oii)
        v3v_dist[(v3v_dist >= v3v_start) & (v3v_dist < voo_start)] = 0
        isect_ii = np.argmin(v3v_dist)
        # Now we just want the closest value in each direction from that point.
        v3v_dist = oii[isect_ii] - oii
        ii = v3v_dist > 0
        if np.sum(ii) == 0:
            raise ValueError(f"no v3v>0 intersections found for "
                             f"{rater}/{sid}/{hemisphere}")
        isect_hv4 = np.where(ii)[0][np.argmin(v3v_dist[ii])]
        ii = v3v_dist < 0
        if np.sum(ii) == 0:
            raise ValueError(f"no v3v<0 intersections found for "
                             f"{rater}/{sid}/{hemisphere}")
        isect_vo = np.where(ii)[0][np.argmax(v3v_dist[ii])]
        # Put these together into the (hii, oii, pts_ho) variables.
        hii = hii[[isect_ii, isect_hv4, isect_vo]]
        oii = oii[[isect_ii, isect_hv4, isect_vo]]
        pts_ho = pts_ho[:,[isect_ii, isect_hv4, isect_vo]]
        ii = np.argsort(hii)
        (hii,oii,pts_ho) = (hii[ii], oii[ii], pts_ho[:,ii])
    elif len(hii) != 3:
        raise RuntimeError(f"{len(hii)} hV4-VO1 / Outer intersections for "
                           f"{rater}/{sid}/{hemisphere}")
    # These intersections will have been sorted by distance along the hV4-VO1
    # boundary, so the first one must necessarily be the "V3-ventral" end.
    hii_v3v = hii[0]
    pii = [1,2] if oii[1] < oii[2] else [2,1]
    (oii_hv4, oii_vo) = oii[pii]
    (hii_hv4, hii_vo) = hii[pii]
    # Shorten the hV4-extended line to just the relevant part.
    if   hii_hv4 < hii_vo: (hii_end, pii_end) = (hii_vo,  pii[1])
    elif hii_hv4 > hii_vo: (hii_end, pii_end) = (hii_hv4, pii[0])
    else:
        x_hv4 = pts_ho[:,pii[0]]
        x_vo  = pts_ho[:,pii[1]]
        x_hii = hv4_vo1[:,hii_hv4]
        if _closer(x_hii, x_hv4, x_vo): (hii_end, pii_end) = (hii_vo,  pii[1])
        else:                           (hii_end, pii_end) = (hii_hv4, pii[0])
    hv4_vo1_norm = np.hstack([pts_ho[:,[0]],
                              hv4_vo1[:, (hii_v3v + 1):hii_end],
                              pts_ho[:,[pii_end]]])
    outer_norm = np.hstack([pts_ho[:, [pii[0]]],
                            outer[:, (oii[pii[0]] + 1):oii[pii[1]]],
                            pts_ho[:, [pii[1]]]])
    # Now, we can find the intersection of outer and VO1/VO2.
    (vii, uii, pts_vu) = _cross_isect(vo1_vo2, outer_norm)
    if len(vii) != 2:
        raise RuntimeError(f"{len(vii)} VO1-VO2 / Outer intersections for "
                           f"{rater}/{sid}/{hemisphere}")
    # If uii is descending, then VO1-VO2 is backwards.
    if uii[1] < uii[0]:
        vo1_vo2 = np.fliplr(vo1_vo2)
        (vii, uii, pts_vu) = _cross_isect(vo1_vo2, outer_norm)
    vo1_vo2_norm = np.hstack([pts_vu[:,[0]], 
                              vo1_vo2[:, (vii[0]+1):(vii[1] + 1)],
                              pts_vu[:,[1]]])
    contours = {'VO1_VO2': vo1_vo2_norm,
                'hV4_VO1': hv4_vo1_norm,
                'outer':   outer_norm}
    # Now make the boundaries.
    hv4_b = np.hstack([
        # First, hV4 from V3-ventral to the hV4-outer.
        pts_ho[:, [0]],
        hv4_vo1[:, (hii_v3v + 1):(hii_hv4 + 1)],
        pts_ho[:, [pii[0]]],
        # Next, outer from this point to back to the hV4-V3-ventral point.
        outer[:, (oii[pii[0]] + 1):oii[0]]])
    vo1_b = np.hstack([
        # First, VO1-VO2.
        vo1_vo2_norm,
        # Then the outer from there to the hV4-VO1 contour.
        outer_norm[:, (uii[1] + 1):],
        # Then the hV4-VO1 boundary.
        pts_ho[:, [pii[1]]],
        np.fliplr(hv4_vo1[:, (hii_v3v + 1):(hii_vo + 1)]),
        pts_ho[:, [0]],
        # Finally, the outer boundary from hV4-VO1 to VO1-VO2.
        outer[:, (oii[0] + 1):(uii[0] + oii[pii[0]] + 1)]])
    vo2_b = np.hstack([
        # First the outer around VO2.
        outer_norm[:, (uii[0] + 1):(uii[1] + 1)],
        # Then, VO1-VO2.
        np.fliplr(vo1_vo2_norm)])
    boundaries = {'hV4': hv4_b, 'VO1': vo1_b, 'VO2': vo2_b}
    return (contours, boundaries)
@pimms.calc('traces')
def calc_traces(flatmap, boundaries, normalized_contours):
    """Calculates path-traces from the flatmap and boundaries.
    """
    mp = flatmap.meta_data['projection']
    traces = {k: ny.path_trace(mp, pts, closed=True)
              for (k,pts) in boundaries.items()}
    for (k,pts) in normalized_contours.items():
        traces[k] = ny.path_trace(mp, pts, closed=False)
    return (traces,)

# The plan ---------------------------------------------------------------------
ventral_contours_plan = pimms.plan(
    init_plan,
    load_v3v_contour=load_v3v_contour,
    extend_contours=calc_extended_contours,
    normalize_contours=calc_normalized_contours,
    traces=calc_traces)
ventral_contours_meanplan = pimms.plan(
    init_meanplan,
    load_v3v_contour=load_v3v_contour,
    extend_contours=calc_extended_contours,
    normalize_contours=calc_normalized_contours,
    traces=calc_traces)
