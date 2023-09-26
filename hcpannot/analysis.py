################################################################################
# analysis.py
#
# Analysis tools for the HCP visual cortex contours.
# by Noah C. Benson <nben@uw.edu>

# Import things
import sys, os, pimms, json, warnings
import numpy as np
import scipy as sp
import pyrsistent as pyr
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import neuropythy as ny

from .core import (image_order, op_flatmap)
from .interface import (
    imgrid_to_flatmap,
    flatmap_to_imgrid,
    subject_data)
from .config import (
    default_imshape,
    subject_list,
    subject_list_1,
    subject_list_2,
    subject_list_3,
    meanrater,
    procdata,
    to_data_path,
    labelkey)
from .io import (
    load_contours)


# Visualization ################################################################

raw_colors = {
    'hV4_outer': (0.5, 0,   0),
    'hV4_VO1':   (0,   0.3, 0),
    'VO_outer':  (0,   0.4, 0.6),
    'VO1_VO2':   (0,   0,   0.5)}
preproc_colors = {
    'hV4_outer': (0.7, 0,   0),
    'hV4_VO1':   (0,   0.5, 0),
    'VO_outer':  (0,   0.6, 0.8),
    'VO1_VO2':   (0,   0,   0.7),
    'V3_ventral':(0.7, 0,   0.7),
    'outer':     (0.7, 0.7, 0, 0)}
ext_colors = {
    'hV4_outer': (1,   0,   0),
    'hV4_VO1':   (0,   0.8, 0),
    'VO_outer':  (0,   0.9, 1),
    'VO1_VO2':   (0,   0,   1),
    'V3_ventral':(0.8, 0,   0.8),
    'outer':     (0.8, 0.8, 0.8, 1)}
boundary_colors = {
    'hV4': (1, 0.5, 0.5),
    'VO1': (0.5, 1, 0.5),
    'VO2': (0.5, 0.5, 1)}

def plot_contours(dat, raw=None, ext=None, preproc=None,
                  contours=None, boundaries=None,
                  figsize=(2,2), dpi=504, axes=None, 
                  flatmap=True, lw=1, color='prf_polar_angle',
                  mask=('prf_variance_explained', 0.05, 1)):
    """Plots a rater's ventral ccontours on the cortical flatmap.

    `plot_contours(data)` plots a flatmap of the visual cortex for the
    subject whose data is contained in the parameter `data`. This parameter must
    be an output dictionary of the `vc_plan` plan. Contours can be drawn on the
    flatmap by providing one or more of the optional arguments `raw`, `ext`,
    `preproc`, `contours`, and `boundaries`. If any of these is set to `True`,
    then that set of contours is drawn on the flatmap with a default
    color-scheme. Alternately, if a dictionary is given, its keys must be the
    contour names and its values must be colors.

    Parameters
    ----------
    data : dict
        An output dictionary from the `vc_plan` plan.
    raw : boolean or dict, optional
        Whether and how to plot the raw contours (i.e., the contours as drawn by
        the raters).
    ext : boolean or dict, optional
        Whether and how to plot the extended raw contours.
    preproc : boolean or dict, optional
        Whether and how to plot the preprocessed contours.
    contours : boolean or dict, optional
        Whether and how to plot the processed contours.
    boundaries : boolean or dict, optional
        Whether and how to plot the final boundaries.
    figsize : tuple of 2 ints, optional
        The size of the figure to create, assuming no `axes` are given. The
        default is `(2,2)`.
    dpi : int, optional
        The number of dots per inch to given the created figure. If `axes` are
        given, then this is ignored. The default is 360.
    axes : matplotlib axes, optional
        The matplotlib axes on which to plot the flatmap and contours. If this
        is `None` (the default), then a figure is created using `figsize` and
        `dpi`.
    flatmap : boolean, optional
        Whether or not to draw the flatmap. The default is `True`.
    lw : int, optional
        The line-width to use when drawing the contours. The default is 1.
    color : str or flatmap property, optional
        The color to use in the flatmap plot; this option is passed directly to
        the `ny.cortex_plot` function. The default is `'prf_polar_angle'`.
    mask : mask-like, optional
        The mask to use when plotting the color on the flatmap. This option is
        passed directly to the `ny.cortex_plot` function. The default value is
        `('prf_variance_explained', 0.05, 1)`.

    Returns
    -------
    matplotlib.Figure
        The figure on which the plot was made.
    """
    # Make the figure.
    if axes is None:
        (fig,ax) = plt.subplots(1,1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(0,0,1,1,0,0)
    else:
        ax = axes
        fig = ax.get_figure()
    # Plot the flatmap.
    if flatmap:
        fmap = dat['flatmap']
        ny.cortex_plot(fmap, color=color, mask=mask, axes=ax)
    # Plot the requested lines:
    if raw is not None:
        if raw is True: raw = raw_colors
        for (k,v) in dat['raw_contours'].items():
            c = raw.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if preproc is not None:
        if preproc is True: preproc = preproc_colors
        for (k,v) in dat['preproc_contours'].items():
            c = preproc.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if ext is not None:
        if ext is True: ext = ext_colors
        for (k,v) in dat['ext_contours'].items():
            c = ext.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if contours is not None:
        if contours is True: contours = ext_colors
        for (k,v) in dat['contours'].items():
            c = contours.get(k, 'w')
            ax.plot(v[0], v[1], '-', color=c, lw=lw)
    if boundaries is not None:
        if boundaries is True: boundaries = boundary_colors
        for (k,v) in dat['boundaries'].items():
            c = boundaries.get(k, 'w')
            x = np.concatenate([v[0], [v[0][0]]])
            y = np.concatenate([v[1], [v[1][0]]])
            ax.plot(x, y, '-', color=c, lw=lw)
    ax.axis('off')
    return fig


# The Watershed Approach #######################################################
# The segmentation algorithm used here was pointed out by Chris Luengo on the
# image processing Stack Exchange (https://dsp.stackexchange.com/users/33605),
# see here for the original implementation:
# https://dsp.stackexchange.com/a/89106/68937

def contours_image(mesh, contours, dpi=512, lw=0.1):
    """Given a mesh and a set of traces, return an image of the traces.
    
    The purpose of this function is to produce an image that can be mapped back
    to the original mesh but that contains the traces drawn in white on a black
    background for use with the watershed algorithm.
    """
    (fig,ax) = plt.subplots(1,1, figsize=(1,1), dpi=dpi)
    fig.subplots_adjust(0,0,1,1,0,0)
    canvas = fig.canvas
    for (x,y) in contours:
        ax.plot(x, y, 'k-', lw=lw)
    (xmin,ymin) = np.min(mesh.coordinates, axis=1)
    (xmax,ymax) = np.max(mesh.coordinates, axis=1)
    ax.axis('off')
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    image = 255 - np.mean(image, -1)
    plt.close(fig)
    return image
def watershed_image(im, fill_contours=True, max_depth=2):
    """Applies the watershed algorithm to an image of contours.
    
    The contours image can be generated with the `contours_image` function. See
    the `watershed_contours` function for information on applying the watershed
    algorithm to the contours themselves.
    """
    import sys, contextlib
    if 'diplib' not in sys.modules:
        # Suppress stdout first time we import.
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                import diplib as dip
    else:
        import diplib as dip
    img = ~dip.Image(im)
    dt = dip.EuclideanDistanceTransform(img, border="object")
    # Ensure image border is a single local maximum
    dip.SetBorder(dt, value=dip.Maximum(dt)[0], sizes=2)
    # Watershed (inverted); the use of maxSize=0 is equivalent to applying
    # an H-Minima transform before applying the watershed. This is the default
    seg = dip.Watershed(
        dt,
        mask=img,
        connectivity=2,
        maxDepth=max_depth,
        flags={"high first", "correct"})
    lbls = np.array(dip.Label(~seg))
    # If requested, we fill in the contours somewhat arbitrarily with values
    # from the neighboring pixels.
    if fill_contours:
        lls = np.unique(lbls)
        if len(lls) < 3:
            raise RuntimeError("watershed produced fewer than 3 labels")
        bg = lbls[0,0]
        lls = [ll for ll in lls if ll != 0 and ll != bg]
        layers = [lbls == ll for ll in lls]
        # Fill in the 0 labels by dilating the inner regions.
        mask = (lbls == 0)
        while mask.any():
            layers = [sp.ndimage.binary_dilation(layer) for layer in layers]
            layernums = [layer[mask]*ll for (ll,layer) in zip(lls, layers)]
            lbls[mask] = np.max(layernums, axis=0)
            mask = (lbls == 0)
        # Extract the background and make it 0.
        if bg == 1:
            lbls -= 1
        elif bg == lls[-1]:
            lbls[lbls == bg] = 0
        else:
            ii = (lbls == bg)
            lbls[lbls > bg] -= 1
            lbls[ii] = 0
    return lbls
def watershed_contours(mesh, contours,
                       dpi=512, lw=0.1,
                       fill_contours=True, max_depth=2):
    """Apply the watershed algorithm to the contours and return mesh labels.
    
    This function uses the watershed algorithm, as implemented in the diplib
    package in order to segment a set of imprecisely-drawn contours. The
    return value is the labels of the mesh vertices. These labels are
    arbitrarily enumerated with the exception that the background is always 0.
    """
    im = contours_image(mesh, contours, dpi=dpi, lw=lw).astype(bool)
    lbls = watershed_image(im, fill_contours=fill_contours, max_depth=max_depth)
    # Invert back to the mesh!
    (xmin,ymin) = np.min(mesh.coordinates, axis=1)
    (xmax,ymax) = np.max(mesh.coordinates, axis=1)
    xpx = (mesh.coordinates[0] - xmin) / (xmax - xmin) * (dpi - 1)
    ypx = (ymax - mesh.coordinates[1]) / (ymax - ymin) * (dpi - 1)
    cs = np.round(xpx).astype(int)
    rs = np.round(ypx).astype(int)
    return lbls[rs, cs]

# The plan for the watershed implementation.
@pimms.calc('watershed_labels')
def calc_watershed_labels(raw_contours, v3v_contour, flatmap):
    """Calculates the labels for the set of contours.
    
    The labels are calculated using a distance-based version of the watershed
    algorithm adapted for use with contour drawings. The segmentation algorithm
    used here was pointed out by Chris Luengo on the image processing Stack
    Exchange (https://dsp.stackexchange.com/users/33605), see here for the
    original implementation: https://dsp.stackexchange.com/a/89106/68937.
    
    Parameters
    ----------
    raw_contours : dict-like
        The dictionary of contours.
    v3v_contour : NumPy array
        The `(2 x N)` matrix of points in the V3-ventral contour.
    flatmap : neuropythy mesh
        The flatmap on which the contours were drawn.
        
    Outputs
    -------
    watershed_labels
        The labels, defined on the mesh vertices, for set of contours. The
        contours are converted into labels using the the watershed
        algorithm. These labels use 0 as the background but are not otherwise
        ordered in any specific way.
    """
    contour_list = list(raw_contours.values()) + [v3v_contour]
    lbls = watershed_contours(flatmap, contour_list)
    lbls.setflags(write=False)
    return (lbls,)
@pimms.calc('labels')
def calc_ventral_labels(watershed_labels, v3v_contour, flatmap, cortex,
                        warn=warnings.warn,
                        label_key=labelkey):
    """Converts the watershed labels into hV4, VO1, and VO2 labels.

    The watershed labels use 0 as the background, but there is no guarantee that
    they use any particular number for each annotated area. This calculation
    ensures that there are the correct number of labels and orders them as
    follows: 1 for hV4, 2 for VO1, and 3 for VO2.

    Parameters
    ----------
    watershed_labels
        The vertex labels for the flatmap, as computed by the watershed
        algorithm, that are to be converted into vertex labels for the cortex
        and ordered with 1 indicating hV4, 2 indicating VO1, and 3 indicating
        VO2.
    v3v_contour
        The `(2 x N)` matrix of points in the V3-ventral contour.
    flatmap
        The flatmap on which the `watershed_labels` were made.
    cortex
        The cortex object used to create `flatmap`.
    label_key
        A dictionary whose keys are the names of the visual areas (`'hV4'`,
        `'VO1'`, and `'VO2'`) and whose values are the integer labels that
        are to be used with each area.

    Outputs
    -------
    labels : numpy vector of ints
        A vector of labels for each vertex on the cortical surface, with 1
        indicating an hV4 vertex, 2 indicating VO1, and 3 indicating VO2. A
        label of 0 indicates that a vertex belongs to none of these areas.
    """
    uniq = np.unique(watershed_labels)
    assert uniq[0] == 0, "watershed_labels does not start with 0"
    nuniq = len(uniq)
    assert nuniq >= 4, \
        f"watershed_labels has less than 4 labels (found {nuniq})"
    assert nuniq <= 4, \
        f"watershed_labels has more than 4 labels (found {nuniq})"
    lbls = watershed_labels
    layers = [lbls == ll for ll in uniq]
    # At this point we have a set of labels that has 4 unique labels; these
    # unique labls may not be sequential, but 0 is the background and uniq is
    # the array of unique label values.
    # We need to figure out which of the labels is hV4, which is VO1, and which
    # is VO2; we can do this by comparing to the start and end positions of
    # the V3v contour and by looking at which labels are neighbors.
    (u,v) = flatmap.tess.indexed_edges
    layers = [np.where(layer)[0] for layer in layers]
    neighbor_lbls = [
        np.unique(
            lbls[np.concatenate([u[np.isin(v, layer)], v[np.isin(u, layer)]])])
        for layer in layers]
    neighbor_lbls = [nl[nl != ll] for (ll,nl) in zip(uniq, neighbor_lbls)]
    assert len(neighbor_lbls[0]) == 3, "background does not touch all labels"
    # One of these will border the other two.
    two_neis = [
        ll for (ll,neis) in zip(uniq, neighbor_lbls)
        if ll != 0 and len(neis) == 3]
    assert len(two_neis) == 1, "hV4 and VO1 are neighbors"
    ll_vo1 = two_neis[0]
    # One of the others will be closest to the v3v tip:
    v3v_xy0 = v3v_contour[:,-1] # the foveal point is the last point here.
    coords = flatmap.coordinates
    v3v_dist = [
        (ll, np.sum((np.mean(coords[:,layer], axis=1) - v3v_xy0)**2))
        for (ll, layer) in zip(uniq, layers)
        if ll != 0 and ll != ll_vo1]
    if v3v_dist[0][1] < v3v_dist[1][1]:
        (ll_hv4, ll_vo2) = (v3v_dist[0][0], v3v_dist[1][0])
    else:
        (ll_hv4, ll_vo2) = (v3v_dist[1][0], v3v_dist[0][0])
    lbls = np.zeros(cortex.vertex_count, dtype=int)
    for (ll, layer) in zip(uniq, layers):
        if   ll == ll_hv4: ll = label_key['hV4']
        elif ll == ll_vo1: ll = label_key['VO1']
        elif ll == ll_vo2: ll = label_key['VO2']
        else: ll = 0
        lbls[flatmap.labels[layer]] = ll
    lbls.setflags(write=False)
    return (lbls,)
@pimms.calc('surface_areas')
def calc_surface_areas(cortex, labels, label_key=labelkey):
    """Calculates the surface area of each visual area.

    This calculation uses the labels of the visual areas to produce surface area
    estimates. These estimates are stored in a dictionary whose keys are named
    in the `label_key` parameter and whose values are in square mm.

    Parameters
    ----------
    cortex
        The cortex object on which surface areas are being calculated.
    labels
        The integer label value for each vertex in `cortex`.
    label_key
        A dictionary whose keys are the names of the visual areas (`'hV4'`,
        `'VO1'`, and `'VO2'`) and whose values are the integer labels that
        are to be used with each area.

    Outputs
    -------
    surface_areas
        A dict-like object of the surface area of each visual area.
    """
    vertex_sarea = cortex.prop('midgray_surface_area')
    surface_areas = {
        k: np.sum(vertex_sarea[labels == ll])
        for (k,ll) in label_key.items()}
    return (surface_areas,)
#plan_ventral_watershed = pimms.plan(
#    parse_chirality=parse_chirality,
#    watershed_labels=calc_watershed_labels,
#    labels=calc_ventral_labels,
#    load_contours=calc_load_contours,
#    load_v3v_contour=load_v3v_contour,
#    load_cortex=load_cortex,
#    flatmap=calc_flatmap,
#    surface_areas=calc_surface_areas)
