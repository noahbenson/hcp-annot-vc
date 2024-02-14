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
    meansid,
    procdata)
from ..io import (
    save_contours, load_contours,
    save_traces,   load_traces,
    save_paths,    load_paths,
    save_labels,   load_labels,
    save_reports,  load_reports)
from .core import (
    init_plan,
    init_plan_meanrater,
    init_plan_meansub)
from .util import (
    cross_isect_2D,
    iscloser,
    fix_polygon,
    contour_endangle,
    order_nearness,
    dedup_points,
    find_crossings,
    extend_contour)


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
@pimms.calc('v3v_contour')
def load_meansub_v3v_contour(contours):
    """Loads the V3-ventral contour from the HCP-lines mean subject.
    
    Parameters
    ----------
    chirality : 'lh' or 'rh'
        The hemisphere (`'lh'` or `'rh'`) to load.
        
    Outputs
    -------
    v3v_contour : NumPy array
        A NumPy array of the points in the V3-ventral contour.
    """
    return (np.fliplr(contours['V3v']),)
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
    (hv4_preproc, vo_preproc, outbound, outer_sources) = _proc_outers(
        contours['hV4_outer'],
        contours['VO_outer'],
        v3v_contour)
    contours['hV4_outer'] = hv4_preproc
    contours['VO_outer'] = vo_preproc
    contours['V3v'] = np.fliplr(v3v_contour)
    contours['outer'] = outbound
    # Now make the extended contours:
    ext_contours = {k:extend_contour(c) for (k,c) in contours.items()}
    # And return!
    return (contours, ext_contours, outer_sources)
def _proc_outers(hv4_outer, vo_outer, v3v_contour):
    # Both the hV4-VO1 contour need to be re-ordered to be starting near
    # the V3-ventral contour and ending far from it. We can also remove
    # duplicate points while we're at it.
    hv4_outer = order_nearness(dedup_points(hv4_outer), v3v_contour[:,-1])
    vo_outer = order_nearness(dedup_points(vo_outer), v3v_contour[:,0])
    # If the two contours have 1 intersection, that is their new endpoint.
    (hii, vii, pts) = cross_isect_2D(hv4_outer, vo_outer)
    while len(hii) > 1:
        # Trim one point off of each end and try again.
        hv4_outer = hv4_outer[:,:-1]
        vo_outer = vo_outer[:,:-1]
        (hii, vii, pts) = cross_isect_2D(hv4_outer, vo_outer)
    if len(hii) == 1:
        # This is their new endpoint.
        hv4_outer = np.hstack([hv4_outer[:, :hii[0]+1], pts])
        vo_outer = np.hstack([vo_outer[:, :vii[0]+1], pts])
    else:
        # Check whether the vector from end to end is pointing in the wrong
        # direction---if so, we need to cut ends off.
        while contour_endangle(hv4_outer, vo_outer[:,-1]) < 0:
            hv4_outer = hv4_outer[:,:-1]
            vo_outer = vo_outer[:,:-1]
        u = 0.5*(hv4_outer[:,-1] + vo_outer[:,-1])
        hv4_outer = np.hstack([hv4_outer, u[:,None]])
        vo_outer = np.hstack([vo_outer, u[:,None]])
    # These can now be turned into an outer boundary that we start and end
    # and the V3v tip.
    outer_bound = np.hstack(
        [np.fliplr(v3v_contour), vo_outer[:, :-1],
         np.fliplr(hv4_outer), v3v_contour[:,[-1]]])
    outer_sources = np.concatenate(
        [['V3v']*v3v_contour.shape[1], ['VO_outer']*(vo_outer.shape[1] - 1),
         ['hV4_outer']*(hv4_outer.shape[1] + 1)])
    return (hv4_outer, vo_outer, outer_bound, outer_sources)
@pimms.calc('normalized_contours', 'boundaries')
def calc_normalized_contours(sid, hemisphere, chirality, rater, outer_sources,
                             preproc_contours, ext_contours):
    """Normalizes the raw contours and converts them into path-traces.
    """
    hv4_vo1 = ext_contours['hV4_VO1']
    vo1_vo2 = ext_contours['VO1_VO2']
    outer = preproc_contours['outer']
    (contours, bounds) = _calc_normcontours_simple(
        hv4_vo1, vo1_vo2, outer, chirality)
    if chirality == 'lh':
        # Make the boundaries counter-clockwise.
        bounds = {k: np.fliplr(v) for (k,v) in bounds.items()}
    bounds = {k: fix_polygon(b) for (k,b) in bounds.items()}
    contours['hV4_outer'] = preproc_contours['hV4_outer']
    contours['VO_outer'] = preproc_contours['VO_outer']
    contours['V3v'] = preproc_contours['V3v']
    return (contours, bounds)
def _calc_normcontours_simple(hv4_vo1, vo1_vo2, outer, chirality):
    # The extended vo1/vo2 and hv4/vo contours should each intersect this outer
    # boundary twice. We subdivide the outer boundary into top and bottom pieces
    # for each of the crossing contours.
    cp0 = -1 if chirality == 'rh' else 1
    crossings = {'hV4-VO1': hv4_vo1, 'VO1-VO2': vo1_vo2}
    outer_work = outer
    pieces = []
    for name in ('hV4-VO1', 'VO1-VO2'):
        cross = crossings[name]
        # First, make sure that cross is pointing in the right direction.
        (cii, oii, pts) = find_crossings(cross, outer)
        if len(cii) != 2:
            raise RuntimeError(
                f"{len(cii)} {name} / Outer intersections")
        if oii[0] > oii[1]:
            cross = np.fliplr(cross)
        # Next, find the crossings with the actual working outer boundary.
        (cii, oii, pts) = find_crossings(cross, outer_work)
        if len(cii) != 2:
            raise RuntimeError(
                f"{len(cii)} {name} / Working-Outer intersections")
        # We can now put together the upper and lower pieces. First, roll the
        # outer_work matrix so that the intersection is between the last two
        # columns of the matrix.
        outer_work = np.roll(outer_work[:,:-1], -oii[0] - 1, axis=1)
        outer_work = np.hstack([outer_work, outer_work[:,[0]]])
        (cii, oii, pts) = find_crossings(cross, outer_work)
        # Now build the pieces.
        cross = np.hstack(
            [pts[:, [0]], cross[:, cii[0]+1:cii[1]+1], pts[:,[1]]])
        crossings[name] = cross
        upper = np.hstack(
            [cross, outer_work[:, oii[1]+1:-1], pts[:,[0]]])
        pieces.append(upper)
        outer_work = np.hstack(
            [pts[:,[0]], outer_work[:, :oii[1]+1], np.fliplr(cross)])
    (hv4_b, vo1_b) = pieces
    vo2_b = outer_work
    hv4_vo1_norm = crossings['hV4-VO1']
    vo1_vo2_norm = crossings['VO1-VO2']
    contours = {'VO1_VO2': vo1_vo2_norm,
                'hV4_VO1': hv4_vo1_norm,
                'outer':   outer}
    bounds = {'hV4': hv4_b, 'VO1': vo1_b, 'VO2': vo2_b}
    return (contours, bounds)
def _calc_normcontours_complex(hv4_vo1, vo1_vo2, outer, outer_sources):
    # We want to find the intersections between hV4-VO1 and the outer boundary.
    # There must be 3; any other number is an error.
    (hii, oii, pts_ho) = cross_isect_2D(hv4_vo1, outer)
    if len(hii) > 3:
        # We need to figure out which three intersections of the hV4-VO1 border
        # and the outer border we're going to use.
        # One of these intersections must be near the V3v border, so we use the
        # outer sources to figure that out.
        if outer_sources is None:
            raise RuntimeError(
                f"{len(hii)} hV4-VO1 / Outer intersections, which have no"
                f" outer_sources data")
        v3v_start = np.where(outer_sources == 'V3v')[0][0]
        voo_start = np.where(outer_sources == 'VO_outer')[0][0]
        v3v_dist = np.abs(voo_start - oii)
        v3v_dist[(v3v_dist >= v3v_start) & (v3v_dist < voo_start)] = 0
        isect_ii = np.argmin(v3v_dist)
        # Now we just want the closest value in each direction from that point.
        v3v_dist = oii[isect_ii] - oii
        ii = v3v_dist > 0
        if np.sum(ii) == 0:
            raise ValueError(f"no v3v>0 intersections found")
        isect_hv4 = np.where(ii)[0][np.argmin(v3v_dist[ii])]
        ii = v3v_dist < 0
        if np.sum(ii) == 0:
            raise ValueError(f"no v3v<0 intersections found")
        isect_vo = np.where(ii)[0][np.argmax(v3v_dist[ii])]
        # Put these together into the (hii, oii, pts_ho) variables.
        hii = hii[[isect_ii, isect_hv4, isect_vo]]
        oii = oii[[isect_ii, isect_hv4, isect_vo]]
        pts_ho = pts_ho[:,[isect_ii, isect_hv4, isect_vo]]
        ii = np.argsort(hii)
        (hii,oii,pts_ho) = (hii[ii], oii[ii], pts_ho[:,ii])
    elif len(hii) != 3:
        raise RuntimeError(f"{len(hii)} hV4-VO1 / Outer intersections")
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
        if iscloser(x_hii, x_hv4, x_vo): (hii_end, pii_end) = (hii_vo,  pii[1])
        else:                           (hii_end, pii_end) = (hii_hv4, pii[0])
    hv4_vo1_norm = np.hstack([pts_ho[:,[0]],
                              hv4_vo1[:, (hii_v3v + 1):hii_end],
                              pts_ho[:,[pii_end]]])
    outer_norm = np.hstack([pts_ho[:, [pii[0]]],
                            outer[:, (oii[pii[0]] + 1):oii[pii[1]]],
                            pts_ho[:, [pii[1]]]])
    # Now, we can find the intersection of outer and VO1/VO2.
    (vii, uii, pts_vu) = cross_isect_2D(vo1_vo2, outer_norm)
    if len(vii) != 2:
        raise RuntimeError(f"{len(vii)} VO1-VO2 / Outer intersections")
    # If uii is descending, then VO1-VO2 is backwards.
    if uii[1] < uii[0]:
        vo1_vo2 = np.fliplr(vo1_vo2)
        (vii, uii, pts_vu) = cross_isect_2D(vo1_vo2, outer_norm)
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
    bounds = {'hV4': hv4_b, 'VO1': vo1_b, 'VO2': vo2_b}
    return (contours, bounds)
@pimms.calc('traces')
def calc_traces(flatmap, boundaries, normalized_contours):
    """Calculates path-traces from the flatmap and boundaries.
    """
    mp = flatmap.meta_data['projection']
    traces = {
        k: ny.path_trace(mp, pts, closed=True)
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
ventral_contours_plan_meanrater = pimms.plan(
    init_plan_meanrater,
    load_v3v_contour=load_v3v_contour,
    extend_contours=calc_extended_contours,
    normalize_contours=calc_normalized_contours,
    traces=calc_traces)
ventral_contours_plan_meansub = pimms.plan(
    init_plan_meansub,
    load_v3v_contour=load_meansub_v3v_contour,
    extend_contours=calc_extended_contours,
    normalize_contours=calc_normalized_contours,
    traces=calc_traces)
