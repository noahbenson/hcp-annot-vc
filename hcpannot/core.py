################################################################################
# Core utilities, types, and routines for the HCP visual annotation project.
# by Noah C. Benson <nben@uw.edu>

import sys, os, pimms
import numpy as np, scipy as sp
import neuropythy as ny
import matplotlib as mpl, matplotlib.pyplot as plt
import ipywidgets as widgets

# Colormaps ####################################################################
highlight_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'highlight',
    [(0.00, (0.5, 0.5, 0.5)),
     (0.49, (0,   0,   0)),
     (0.50, (1,   0,   0)),
     (0.51, (1,   1,   1)),
     (1.00, (0.5, 0.5, 0.5))])
highpeak_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'highpeak',
    [(0.00,  (0.6, 0.6, 0.6)),
     (0.50,  (0.5, 0.5, 0.5)),
     (0.65,  (0,   0,   0)),
     (0.70,  (1,   0,   0)),
     (0.75,  (1,   1,   0)),
     (0.80,  (1,   1,   1)),
     (0.85,  (0.7, 0.7, 0.7)),
     (1.00,  (0.6, 0.6, 0.6))])
image_order = ('curvature',
               'polar_angle',
               'eccentricity',
               'isoang_90',
               'isoang_hml',
               'isoang_hmu',
               'isoang_vml',
               'isoang_vmu',
               'isoecc_0.5',
               'isoecc_1',
               'isoecc_2',
               'isoecc_4',
               'isoecc_7',
               'eccpeak_6.25',
               'thresh_polar_angle',
               'thresh_eccentricity',
               'thresh_isoang_90',
               'thresh_isoang_hml',
               'thresh_isoang_hmu',
               'thresh_isoang_vml',
               'thresh_isoang_vmu',
               'thresh_isoecc_0.5',
               'thresh_isoecc_1',
               'thresh_isoecc_2',
               'thresh_isoecc_4',
               'thresh_isoecc_7')

# Utilities ####################################################################
def op_flatmap(hem, **kw):
    '''
    op_flatmap(hem) yields a flatmap of given hemisphere that places the
      subject's occipital pole of the hemisphere at the center of the map and
      the subject's right on the right side of the map. The radius is pi/2 by
      default, but optional arguments for the mask_flatmap function may be
      provided.      
    '''
    opts = dict(radius=np.pi/2, map_right='right')
    for (k,v) in kw.items():
        opts[k] = v
    return hem.mask_flatmap(('Destrieux09_parcellation', 43), **opts)
def nestget(d, k):
    """Retrieves nested data from the proc dictionaries.
    
    Certain keys such as `'boundaries'` are accessible in the dictionaries that
    are returned by the `proc` function only via the `'nested_data'` key, which
    typically contains another proc dictionary with additional data. The
    `nestget` function gets data from these embedded dictionaries.
    """
    while k not in d:
        d = d['nested_data']
    return d[k]

# Plot Figures #################################################################
def plot_isoecc(fmap, ecc0, scale=1, log=True, cmap='highlight', axes=None,
                cod_threshold=None):
    ecc = fmap.prop('prf_eccentricity')
    if log:
        de = ny.to_logeccen(ecc) - ny.to_logeccen(ecc0)
        q = (de + 1) * 0.5
    else:
        de = (ecc - ecc0)
        q = 1 / (1 + np.exp(-scale*de))
    if cmap == 'highlight': cmap = highlight_cmap
    elif cmap == 'highpeak': cmap = highpeak_cmap
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color=q, mask=mask,
                          vmin=0, vmax=1, cmap=cmap)
def plot_isoang(fmap, ang0, axes=None, scale=180, abs=True, cmap='highlight',
                cod_threshold=None):
    ang = fmap.prop('prf_polar_angle')
    if abs: (ang,ang0) = (np.abs(ang), np.abs(ang0))
    da = (ang - ang0)
    #q = 1 / (1 + np.exp(-scale*da))
    q = da / (2*scale) + 0.5
    q[ny.util.nangt(q, 1)] = 1
    q[ny.util.nanlt(q, 0)] = 0
    if cmap == 'highlight': cmap = highlight_cmap
    elif cmap == 'highpeak': cmap = highpeak_cmap
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color=q, mask=mask,
                          vmin=0, vmax=1, cmap=cmap)
def plot_eccpeak(fmap, ecc0, cmap='highpeak', axes=None, cod_threshold=None):
    ecc = fmap.prop('prf_eccentricity')
    # Transform for the colormap.
    ecc = (np.log(ecc + 0.75) - np.log(0.75)) / (np.log(ecc0) - np.log(0.75))
    if cmap == 'highlight': cmap = highlight_cmap
    elif cmap == 'highpeak': cmap = highpeak_cmap
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color=ecc, mask=mask,
                          vmin=0, vmax=1, cmap=cmap)
def plot_vmbound(fmap, vd, axes=None, cmap='highpeak', cod_threshold=None):
    ang = np.array(fmap.prop('prf_polar_angle'))
    ii = np.isfinite(ang)
    ang[ii] = np.mod(ang[ii] + 180, 360) - 180
    h = fmap.meta_data['projection'].chirality
    if h == 'rh': ang = -ang
    vd = vd[0].lower()
    if vd == 'd':
        # Flip up and down
        ang[ii] = np.sign(ang[ii])*(180 - np.abs(ang[ii]))
    # Center at +x axis
    ang[ii] = np.mod(90 - ang[ii] + 180, 360) - 180
    if cmap == 'highlight': cmap = highlight_cmap
    elif cmap == 'highpeak': cmap = highpeak_cmap
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color=ang, mask=mask,
                          vmin=-180, vmax=180, cmap=cmap)
def plot_hmbound(fmap, vd, axes=None, cmap='highpeak', cod_threshold=None):
    ang = np.array(fmap.prop('prf_polar_angle'))
    ii = np.isfinite(ang)
    ang[ii] = np.mod(ang[ii] + 180, 360) - 180
    h = fmap.meta_data['projection'].chirality
    if h == 'rh': ang = -ang
    vd = vd[0].lower()
    if vd == 'd':
        # Flip up and down
        ang = np.sign(ang)*(180 - np.abs(ang))
    # Center at +x axis
    if cmap == 'highlight': cmap = highlight_cmap
    elif cmap == 'highpeak': cmap = highpeak_cmap
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color=ang, mask=mask,
                          vmin=-180, vmax=180, cmap=cmap)
def plot_curv(fmap, axes=None, cmap='gray', prop='convexity'):
    s = fmap.property(prop)
    s = s - np.min(s)
    s /= np.max(s)
    return ny.cortex_plot(fmap, axes=axes, color=s, vmin=1, vmax=0, cmap=cmap)
def plot_angle(fmap, axes=None, cod_threshold=None):
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color='prf_polar_angle', mask=mask)
def plot_eccen(fmap, axes=None, cod_threshold=None):
    mask=(None if cod_threshold is None else
          ('prf_variance_explained', cod_threshold, np.inf))
    return ny.cortex_plot(fmap, axes=axes, color='prf_eccentricity', mask=mask)
def plot_as_image(plotfn, fmap, *args, **kw):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3,3), dpi=72*4)
    fig.subplots_adjust(0,0,1,1,0,0)
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    kw['axes'] = ax
    plotfn(fmap, *args, **kw)
    (x,y) = fmap.coordinates
    (xmin,xmax) = (np.min(x), np.max(x))
    (ymin,ymax) = (np.min(y), np.max(y))
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.axis('off')
    canvas.draw()       # draw the canvas, cache the renderer
    im = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plt.close(fig)
    n2 = len(im) // 3
    n = np.sqrt(n2)
    im = np.reshape(im, (int(n), int(n), 3))
    return im
def generate_images(fmap, cod_threshold=0.1):
    '''
    generate_images(fmap) yields a dictionary of images of the given flatmap
    that can be used with line-drawing.
    '''
    import pimms
    from neuropythy.util import curry
    h = fmap.meta_data['projection'].chirality
    ppi = plot_as_image
    ims = dict(
        curvature=curry(ppi, plot_curv, fmap),
        polar_angle=curry(ppi, plot_angle, fmap),
        eccentricity=curry(ppi, plot_eccen, fmap),
        thresh_polar_angle=curry(ppi, plot_angle, fmap,
                                 cod_threshold=cod_threshold),
        thresh_eccentricity=curry(ppi, plot_eccen, fmap,
                                  cod_threshold=cod_threshold))
    ims['isoecc_0.5'] = curry(ppi, plot_isoecc, fmap, 0.5)
    ims['thresh_isoecc_0.5'] = curry(ppi, plot_isoecc, fmap, 0.5,
                                     cod_threshold=cod_threshold)
    for ecc in [1,2,4,7]:
        ims['isoecc_%d'%ecc] = curry(ppi, plot_isoecc, fmap, ecc)
        ims['thresh_isoecc_%d'%ecc] = curry(ppi, plot_isoecc, fmap, ecc,
                                             cod_threshold=cod_threshold)
    # The eccen peak image for the hV4/VO1 eccentricity reversal
    ims['eccpeak_6.25'] = curry(ppi, plot_eccpeak, fmap, 6.25)
    ims['thresh_eccpeak_6.25'] = curry(ppi, plot_eccpeak, fmap, 6.25,
                                       cod_threshold=cod_threshold)
    ims['isoang_90'] = curry(ppi, plot_isoang, fmap, 90, abs=True)
    ims['isoang_vmu'] = curry(ppi, plot_vmbound, fmap, 'v')
    ims['isoang_vml'] = curry(ppi, plot_vmbound, fmap, 'd')
    ims['isoang_hmu'] = curry(ppi, plot_hmbound, fmap, 'v')
    ims['isoang_hml'] = curry(ppi, plot_hmbound, fmap, 'd')
    ims['thresh_isoang_90'] = curry(ppi, plot_isoang, fmap, 90, abs=True,
                                    cod_threshold=cod_threshold)
    ims['thresh_isoang_vmu'] = curry(ppi, plot_vmbound, fmap, 'v',
                                     cod_threshold=cod_threshold)
    ims['thresh_isoang_vml'] = curry(ppi, plot_vmbound, fmap, 'd',
                                     cod_threshold=cod_threshold)
    ims['thresh_isoang_hmu'] = curry(ppi, plot_hmbound, fmap, 'v',
                                     cod_threshold=cod_threshold)
    ims['thresh_isoang_hml'] = curry(ppi, plot_hmbound, fmap, 'd',
                                     cod_threshold=cod_threshold)
    return pimms.lmap(ims)
def label_to_segs(fmap, prop, mask=Ellipsis):
    '''
    Given a flatmap and a property that represents labels on the flatmap, yields
      a set of line segments that make up the boundaries between the labels.
    '''
    p = fmap.prop(prop)
    if mask is Ellipsis:
        ii = np.where(np.isfinite(p))[0]
    elif mask is None:
        ii = fmap.tess.indices
    else:
        ii = fmap.mask(mask)
    abc = fmap.tess.indexed_faces
    xy = fmap.coordinates
    ff = np.where(np.all(np.isin(abc, ii), axis=0))[0]
    (pa,pb,pc) = [p[kk] for kk in abc[:,ff]]
    (abeq,bceq,caeq) = (pa == pb, pb == pc, pc == pa) 
    abceq = np.sum([abeq, bceq, caeq], axis=0, dtype='int')
    ff0 = ff[abceq == 0]
    abc1 = abceq == 1
    # faces that are all different we do first:
    (a,b,c) = [xy[:,ii] for ii in abc[:,ff0]]
    ab = np.mean((a,b), axis=0)
    bc = np.mean((b,c), axis=0)
    ca = np.mean((c,a), axis=0)
    mu = np.mean((a,b,c), axis=0)
    segs0 = np.transpose(
        [np.concatenate([mu.T, mu.T, mu.T], axis=0),
         np.concatenate([bc.T, ca.T, ab.T], axis=0)],
        (1,0,2))
    # Next the ones that cross:
    (abeq, bceq, caeq) = [(u & abc1) for u in [abeq,bceq,caeq]]
    (a,b,c) = [xy[:,ii] for ii in abc]
    (ab,bc,ca) = [u.T for u in np.mean([[a,b],[b,c],[c,a]], axis=1)]
    segs1_c = np.transpose([bc[abeq], ca[abeq]], (1,0,2))
    segs1_b = np.transpose([ab[caeq], bc[caeq]], (1,0,2))
    segs1_a = np.transpose([ca[bceq], ab[bceq]], (1,0,2))
    segs1 = np.concatenate([segs1_a, segs1_b, segs1_c], axis=0)
    segs = np.concatenate(
        [segs0, segs1],
        axis=0)
    return segs
