################################################################################
# interface.py
#
# Interface for drawing the HCP lines.
# by Noah C. Benson <nben@uw.edu>

# Import things
import sys, os, pimms
import numpy as np
import pyrsistent as pyr
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import neuropythy as ny

from .core import image_order
from .config import (
    default_imshape,
    default_xlim,
    default_ylim,
    default_grid,
    default_load_path,
    default_osf_url,
    subject_list)

def imgrid_to_flatmap(pts,
                      grid=None,
                      imshape=None,
                      xlim=None,
                      ylim=None):
    '''
    `imgrid_to_flatmap(pts)` yields a 2xN matrix the same size as the given
      (2xN) matrix `pts`, for which the points have been converted from
      coordinates in the given image grid (`grid` option).
    '''
    if grid is None: grid = default_grid
    if imshape is None: imshape = default_imshape
    if xlim is None: xlim = default_xlim
    if ylim is None: ylim = default_ylim
    pts = np.array(pts)
    (R,C) = imshape[:2]
    rg = len(grid)
    cg = len(grid[0])
    (rs,cs) = (R/rg, C/cg)
    rmu = (rs-1)/2
    cmu = (cs-1)/2
    (xmin,xmax) = xlim
    (ymin,ymax) = ylim
    xmu = 0.5*(xmin + xmax)
    ymu = 0.5*(ymin + ymax)
    rpx2yu = -(ymax - ymin) / rs
    cpx2xu = (xmax - xmin) / cs
    (c,r) = pts if pts.shape[0] == 2 else pts.T
    c = np.array(c)
    r = np.array(r)
    while True:
        ii = c > cs
        if not ii.any(): break
        c[ii] -= cs
    while True:
        ii = r > rs
        if not ii.any(): break
        r[ii] -= rs
    x = xmu + (c - cmu)*cpx2xu
    y = ymu + (r - rmu)*rpx2yu
    return np.array([x,y])
def flatmap_to_imgrid(pts,
                      grid=None,
                      imshape=None,
                      xlim=None,
                      ylim=None):
    '''
    `flatmap_to_imgrid(pts)` yields a 2xN matrix the same size as the given
      (2xN) matrix `pts`, for which the points have been converted from
      coordinates in the default flatmap representation to the given
      image grid (`grid` option).
    '''
    if grid is None: grid = default_grid
    if imshape is None: imshape = default_imshape
    if xlim is None: xlim = default_xlim
    if ylim is None: ylim = default_ylim
    pts = np.asarray(pts)
    (R,C) = imshape[:2]
    rg = len(grid)
    cg = len(grid[0])
    (rs,cs) = (R/rg, C/cg)
    rmu = (rs-1)/2
    cmu = (cs-1)/2
    (xmin,xmax) = xlim
    (ymin,ymax) = ylim
    xmu = 0.5*(xmin + xmax)
    ymu = 0.5*(ymin + ymax)
    yu2rpx = -rs / (ymax - ymin)
    xu2cpx = cs / (xmax - xmin)
    (x,y) = pts if pts.shape[0] == 2 else pts.T
    c = cmu + (x - xmu)*xu2cpx
    r = rmu + (y - ymu)*yu2rpx
    return np.array([[[c+c0*cs,r+r0*rs] for (c0,_) in enumerate(g)]
                     for (r0,g) in enumerate(grid)])

def point_decorate_plot(ax, pts, *args, **kw):
    grid = kw.pop('grid', default_grid)
    imshape = kw.pop('imshape', default_imshape)
    xlim = kw.pop('xlim', default_xlim)
    ylim = kw.pop('ylim', default_ylim)
    rcs = flatmap_to_imgrid(pts, grid=grid, imshape=imshape, xlim=xlim, ylim=ylim)
    plots = [ax.plot(c, r, *args, **kw)
             for row in rcs
             for (r,c) in row]
    return plots
def segs_decorate_plot(ax, segs, *args, **kw):
    from matplotlib.collections import LineCollection as lncol
    grid = kw.pop('grid', default_grid)
    imshape = kw.pop('imshape', default_imshape)
    xlim = kw.pop('xlim', default_xlim)
    ylim = kw.pop('ylim', default_ylim)
    if isinstance(segs, dict):
        tmp = [np.asarray(u) for u in segs.values()]
        tmp = [np.reshape(np.transpose([u[:,:-1], u[:,1:]], (2,0,1)), (-1,2)) for u in tmp]
        pts_nx2 = np.vstack(tmp)
    else:
        pts_nx2 = np.reshape(segs, (-1, 2)).T
    rcs = flatmap_to_imgrid(pts_nx2, grid=grid, imshape=imshape,
                            xlim=xlim, ylim=ylim)
    plots = [lncol(segs, *args, **kw)
             for row in rcs
             for segs0 in row
             for segs in [np.reshape(segs0.T, (-1,2,2))]]
    for p in plots:
        ax.add_collection(p)
    return plots
def clicks_decorate_plot(ax, pts, *args, **kw):
    grid = kw.pop('grid', default_grid)
    imshape = kw.pop('imshape', default_imshape)
    xlim = kw.pop('xlim', default_xlim)
    ylim = kw.pop('ylim', default_ylim)
    (rs,cs) = imshape[:2]
    rs /= len(grid)
    cs /= len(grid[0])
    if len(pts) > 0:
        (x,y) = np.transpose(pts)
    else:
        (x,y) = ([], [])
    x = np.mod(x, cs)
    y = np.mod(y, rs)
    plots = []
    for (r,row) in enumerate(grid):
        for (c,col) in enumerate(row):
            pp = ax.plot(x + cs*c, y + rs*r, *args, **kw)
            for p in pp: plots.append(p)
    return plots
def clicks_update_plot(ax, plots, pts, grid=None, imshape=None):
    if grid is None: grid = default_grid
    if imshape is None: imshape = default_imshape
    (rs,cs) = imshape[:2]
    rs /= len(grid)
    cs /= len(grid[0])
    (x,y) = np.transpose(pts)
    x = np.mod(x, cs)
    y = np.mod(y, rs)
    for plot in plots:
        (px, py) = plot.get_data()
        if len(px) > 0:
            dx = px[0] - x[0]
            dy = py[0] - y[0]
            plot.set_data(x+dx, y+dy)
    return plots

# Functions for loading data. ##################################################
def load_sub_v123(sid, load_path=None, osf_url=None):
    if load_path is None: load_path = default_load_path
    if osf_url is None: osf_url = default_osf_url
    path = ny.util.pseudo_path(osf_url, cache_path=load_path)
    path = path.local_path('annot-v123', '%d.json.gz' % (sid,))
    return ny.load(path)
def load_sub_csulc(sid, load_path=None, osf_url=None):
    if load_path is None: load_path = default_load_path
    if osf_url is None: osf_url = default_osf_url
    path = ny.util.pseudo_path(osf_url, cache_path=load_path)
    path = path.local_path('annot-csulc', '%d.json.gz' % (sid,))
    return ny.load(path)
def load_subimage(sid, h, name, load_path=None):
    from PIL import Image
    if load_path is None: load_path = default_load_path
    flnm = os.path.join(load_path, 'annot-images', str(sid),
                        '%d_%s_%s.png' % (sid, h, name))
    with Image.open(flnm) as im:
        arr = np.array(im)
    return arr
def curry_load_subimage(sid, h, name, load_path=None):
    return lambda:load_subimage(sid, h, name, load_path=load_path)
def load_subwang(sid, h, load_path=None):
    import neuropythy as ny
    if load_path is None: load_path = default_load_path
    flnm = os.path.join(load_path, 'annot-images', str(sid),
                        '%d_%s_wang.mgz' % (sid, h))
    return np.array(ny.load(flnm, 'mgh', to='data'))
def imcat(grid):
    col = [np.concatenate(row, axis=1) for row in grid]
    return np.concatenate(col, axis=0)
def plot_imcat(ims, grid, k):
    grid = [[ims.get(k if g is None else g) for g in row]
            for row in grid]
    dflt = (default_imshape[0]//2, default_imshape[1]//2, 3)
    grid = [[(np.ones(dflt, dtype='int')*255 if g is None else g) for g in row]
            for row in grid]
    return imcat(grid)
# We can (lazily) load the V1-V3 contours now (we could altrnately load them in
# prep_subdata() function, but this prevents them from being loaded once for
# each hemisphere).
v123_contours = pimms.lmap({s: ny.util.curry(load_sub_v123, s)
                            for s in subject_list})
csulc_contours = pimms.lmap({s: ny.util.curry(load_sub_csulc, s)
                            for s in subject_list})
def load_subdata(sid, h, load_path=None, osf_url=None):
    if load_path is None: load_path = default_load_path
    if osf_url is None: osf_url = default_osf_url
    dirname = os.path.join(load_path, 'annot-images', str(sid))
    if not os.path.isdir(dirname):
        pp = ny.util.pseudo_path(osf_url)
        path = pp.local_path('annot-images', '%d.tar.gz' % sid)
        outpath = os.path.join(load_path, 'annot-images')
        import tarfile
        with tarfile.open(path) as fl:
            fl.extractall(outpath)
        # We can go ahead and delete that tarball after we have extracted it;
        # it's just a temporary file anyway.
        os.remove(path)
    ims = {imname: curry_load_subimage(sid, h, imname, load_path=load_path)
           for imname in image_order}
    ims['wang'] = lambda:load_subwang(sid, h, load_path=load_path)
    ims['v123'] = lambda:v123_contours[sid][h]
    ims['csulc'] = lambda:csulc_contours[sid][h]
    return pimms.lmap(ims)
def curry_load_subdata(sid, h, load_path=None, osf_url=None):
    return lambda:load_subdata(sid, h, load_path=load_path, osf_url=osf_url)
def prep_subdata(load_path=None, subject_list=subject_list, osf_url=None):
    return pimms.lmap({(sid,h): curry_load_subdata(sid, h, load_path, osf_url)
                       for sid in subject_list
                       for h in ['lh','rh']})
subject_data = prep_subdata()

# Drawn Contour Data
contour_data = [
    # Ventral Contours: hV4, VO1, VO2:
    dict(name='hV4/VO1 Boundary', save='hV4_VO1', legend='hV4_VO1',
         image='eccpeak_6.25'),
    dict(name='hV4/VO1 Middle (*)', save='hV4_VO1_mid', legend='isoang_hV4_VO1_mid',
         image='isoang_90', optional=True),
    dict(name='hV4 Ventral Boundary', save='hV4', legend='hV4_ventral',
         image='isoang_vml', start=('end', 'V3_ventral')),
    dict(name='V3 Ventral Extension (*)', save='V3v_ext', legend='V3v_ext',
         image='isoang_vmu', optional=True),
    dict(name='VO1/VO2 Interior Boundary', save='VO1_VO2', legend='VO1_VO2',
         image='isoang_vmu'),
    dict(name='VO1+VO2 Outer Boundary', save='VO_outer', legend='VO_outer',
         image='isoang_vml', start=('start', 'V3_ventral')),
    # Dorsal Contours: V3A/B, IPS0, LO1
    dict(name='V3A/B Outer Boundary', save='V3ab_outer', legend='V3ab_outer',
         image='isoang_vmu', start=('end', 'V3_dorsal')),
    dict(name='V3A/B Inner Boundary', save='V3ab_inner', legend='V3ab_inner',
         image='isoecc_0.5'),
    dict(name='IPS0 Outer Boundary', save='IPS0_outer', legend='ips0_outer',
         image='isoang_vml'),
    dict(name='LO1 Outer Boundary', save='LO1_outer', legend='LO1_outer',
         image='isoang_vmu', start=('end', 'V3_ventral')),
]
contours = {cd['name']: cd for cd in contour_data}
default_start_contour = contour_data[0]['name']

# Cortical Sulcus data for plotting cotical sulc boundaries.
csulc_labels = [
    (2, 'Inferior Occipital Gyrus', 'IOG',             [ 0.5,  0.5,    1]),
    (3, 'Fusiform Gyrus', 'FG',                        [   0,    0,    1]),
    (6, 'Calcarine Sulcus', 'CaS',                     [   1,    1,    1]),
    (7, 'Occipito-Temporal Sulcus', 'OTS',             [ 0.5,    1,  0.5]),
    (8, 'Mid-Fusiform Sulcus', 'mFS',                  [   0,    0,  0.5]),
    (13, 'Intra-Parietal Sulcus', 'IPS',               [0.25,    0,    0]),
    (15, 'Posterior Superior Temporal Sulcus', 'pSTS', [   1,    1, 0.25]),
    (16, 'Posterior Lingual Sulcus', 'PLS',            [0.25,    0, 0.25]),
    (19, 'Superior Temporal Sulcus', 'STS',            [   1,  0.5,    0]),
    (21, 'Anterior Lingual Sulcus', 'ALS',             [ 0.5,    0,  0.5]),
    (23, 'Lateral Occipital Sulcus', 'LOS',            [   0,    0,    0]),
    (24, 'Transverse Occipital Sulcus', 'TOS',         [   1,  0.5,  0.5]),
    (25, 'Collateral Sulcus', 'CoS',                   [   1,    0,    1]),
    (26, 'Inferior Temporal Sulcus', 'ITS',            [   0,  0.5,    0]),
    (27, 'Posterior Collateral Sulcus', 'ptCoS',       [   1, 0.75,    1]),
    (28, 'Parietal-Occipital Sulcus', 'POS',           [   1,    0,    0])
]
csulc_abbrevs = {lbl: abbrev for (lbl,nm,abbrev,clr) in csulc_labels}
csulc_numbers = {abbrev: lbl for (lbl,nm,abbrev,clr) in csulc_labels}
csulc_colors  = {abbrev: clr for (lbl,nm,abbrev,clr) in csulc_labels}

# Legend loading/prep.
def load_legimage(load_path, h, imname):
    from PIL import Image
    flname = os.path.join(load_path, 'legends', f'{h}_{imname}.png')
    if not os.path.isfile(flname): return None
    with Image.open(flname) as im:
        arr = np.array(im)
        ii = arr == 255
        arr[np.all(ii, axis=-1), :] = 0
    return arr
def curry_load_legimage(load_path, h, imname):
    return lambda:load_legimage(load_path, h, imname)
def prep_legends(load_path=None, osf_url=None):
    if load_path is None: load_path = default_load_path
    if osf_url is None: osf_url = default_osf_url
    dirname = os.path.join(load_path, 'legends')
    if not os.path.isdir(dirname):
        pp = ny.util.pseudo_path(osf_url)
        path = pp.local_path('annot-images', 'legends.tar.gz')
        import tarfile
        with tarfile.open(path) as fl:
            fl.extractall(load_path)
    ims = {
        h: pimms.lmap(
            {cd['name']: curry_load_legimage(load_path, h, cd['legend'])
             for cd in contours.values()
             if 'legend' in cd})
        for h in ['lh','rh']}
    return pyr.pmap(ims)
legend_data = prep_legends()


# #ROITool #####################################################################
class ROITool(object):
    '''
    ROITool is a tool for drawing ROIs and contours on the HCP.
    '''
    def __init__(self,
                 figsize=1, sidepanel_width='250px', dropdown_width='85%',
                 savedir=None,
                 start_contour=None,
                 grid=None, dpi=72*8,
                 contour_lw=0.25, contour_ms=0.25,
                 csulc_lw=0.33):
        from neuropythy.util import curry
        # Parse default arguments.
        if start_contour is None: start_contour = default_start_contour
        if grid is None: grid = default_grid
        # Copy over the simple parameters of the class first.
        self.grid = grid
        self.start_contour = start_contour
        self.contour_lw = contour_lw
        self.contour_ms = contour_ms
        self.csulc_lw = csulc_lw
        # Parse a few arguments.
        if savedir is None:
            savedir = os.environ.get('GIT_USERNAME', None)
        if savedir is None:
            raise ValueError(
                'Please provide a save directory (savedir option)')
        savedir = os.path.join('/', 'save', savedir)
        savedir = os.path.expanduser(savedir)
        if not os.path.isdir(savedir):
            os.makedirs(savedir, mode=0o755)
        self.savedir = savedir
        # We need to load up the clicks if there are any saved.
        self.clicks = None
        self.clicks_updated = {}
        self.load_clicks()
        start_cd = contours[start_contour]
        (grid_rs, grid_cs) = (len(grid), len(grid[0]))
        figh = figsize * grid_rs / grid_cs
        disp_layout = {'width': "90%",
                       'display': 'flex',
                       'flex-direction': 'row',
                       'justify_content': 'flex-start'}
        dispbox_layout = {'width': "90%",
                          'display': 'flex',
                          'flex-direction': 'row',
                          'justify_content': 'flex-start'}
        dispinn_layout = {'width': "100%",
                          'display': 'flex',
                          'flex-direction': 'row',
                          'align_items': 'flex-end'}
        # Go ahead and setup all the Widgets.
        # Subject (SID) selection:
        self.sid_select = widgets.Dropdown(
            options=subject_list,
            value=subject_list[0],
            description='SID:',
            layout={'width': dropdown_width})
        # Hemisphere (LH/RH) selection:
        self.hemi_select = widgets.Dropdown(
            options=['LH','RH'],
            value='LH',
            description='Hemi:',
            layout={'width': dropdown_width})
        # Contour selection:
        self.contour_select = widgets.Dropdown(
            options=list(contours.keys()),
            value=self.start_contour,
            description='Contour:',
            layout={'width': dropdown_width})
        # Whether to show the Wang lines:
        self.wang_shown = widgets.Checkbox(
            description='Show Contours?',
            value=False,
            indent=False,
            layout=disp_layout)
        # What color to use for the Wang lines:
        self.wang_color = widgets.ColorPicker(
            #description='Wang Color:',
            concise=False,
            value='yellow',
            layout=disp_layout)
        self.wang_disp_box = widgets.VBox(
            (widgets.Label("Wang Atlas:"),
             widgets.VBox((self.wang_shown, self.wang_color),
                          layout=dispinn_layout)),
            layout=dispbox_layout)
        # Whether to show the V1-V3 lines:
        self.v123_shown = widgets.Checkbox(
            description='Show Contours?',
            value=True,
            indent=False,
            layout=disp_layout)
        # What color to use for the Wang lines:
        self.v123_color = widgets.ColorPicker(
            #description='Expert V1-V3 Color:',
            concise=False,
            value='white',
            layout=disp_layout)
        self.v123_disp_box = widgets.VBox(
            (widgets.Label("Expert V1-V3 Color:"),
             widgets.VBox((self.v123_shown, self.v123_color),
                          layout=dispinn_layout)),
            layout=dispbox_layout)
        # Whether to show the already-drawn contours?
        self.work_shown = widgets.Checkbox(
            description='Show Contours?',
            value=True,
            indent=False,
            layout=disp_layout)
        # What color to show the already-drawn contours?
        self.work_color = widgets.ColorPicker(
            #description='Contours Color:',
            concise=False,
            value='#01A9DB',
            layout=disp_layout)
        self.work_disp_box = widgets.VBox(
            (widgets.Label("Drawn Contours Color:"),
             widgets.VBox((self.work_shown, self.work_color),
                          layout=dispinn_layout)),
            layout=dispbox_layout)
        # What color to show the already-drawn contours?
        self.draw_color = widgets.ColorPicker(
            #description='Draw Color:',
            concise=False,
            value='cyan',
            layout=disp_layout)
        self.draw_disp_box = widgets.VBox(
            (widgets.Label("Current Contour Color:"),
             widgets.VBox((self.draw_color,),
                          layout=dispinn_layout)),
            layout=dispbox_layout)
        # The notes section.
        self.notes_area = widgets.Textarea(
            value='', 
            description='',
            layout={'width': '95%', 'height': sidepanel_width})
        # The panel containing the notes section.
        self.notes_panel = widgets.VBox(
            [widgets.Label('Contour Notes:'), self.notes_area],
            layout={'align_items': 'flex-start', 'width':'100%'})
        # The save and reset buttons:
        self.save_button = widgets.Button(description='Save')
        self.reset_button = widgets.Button(description='Reset')
        self.save_box = widgets.HBox(
            children=[self.save_button, self.reset_button],
            layout={'align_items': 'center'})
        # Whether to show the each csulc contour's lines:
        self.csulc_showns = {
            abbrev: widgets.Checkbox(
                description=f'Show {abbrev}?',
                value=False,
                indent=False,
                layout=disp_layout)
            for (_,_,abbrev,_) in csulc_labels}
        # What color to use for the Wang lines:
        self.csulc_colors = {
            abbrev: widgets.ColorPicker(
                concise=False,
                value=mpl.colors.to_hex(clr),
                layout=disp_layout)
            for (_,_,abbrev,clr) in csulc_labels}
        self.csulc_disp_boxes = {
            abbrev: widgets.VBox(
                (widgets.Label(f"{name}:"),
                 widgets.VBox([shown, color], layout=dispinn_layout)),
                layout=(dispbox_layout | {'min_height': '96px'}))
            for ((_,name,abbrev,_), shown, color) \
            in zip(csulc_labels,
                   self.csulc_showns.values(),
                   self.csulc_colors.values())}
        # These are tuples of all the objects that have an influence on the
        # display of the widgets. They are sorted by tabs in the control panel.
        self.controls_select = (self.sid_select,
                                self.hemi_select,
                                self.contour_select,
                                self.notes_panel,
                                self.save_button,
                                self.reset_button)
        self.controls_display = (self.draw_disp_box,
                                 self.work_disp_box,
                                 self.v123_disp_box,
                                 self.wang_disp_box)
        # One more tab for the cortical sulc outlines:
        self.all_csulc_button = widgets.Button(description='All')
        self.none_csulc_button = widgets.Button(description='None')
        self.controls_csulc = ((self.all_csulc_button, self.none_csulc_button) +
                               tuple(self.csulc_disp_boxes.values()))
        self.controls = (self.controls_select +
                         self.controls_display +
                         self.controls_csulc)
        # Go ahead and make the control panel for both the selection and the
        # display tabs.
        control_layout = {'height': f"{figh*dpi*0.65}px",
                          'width': sidepanel_width,
                          'display': 'flex',
                          'flex_flow': 'column',
                          'flex_wrap': 'nowrap',
                          'align_items': 'center',
                          'justify_content': 'flex-start'}
        csulc_layout = {'overflow_y':'auto', 'display':'block'} | control_layout
        self.select_panel = widgets.Box(self.controls_select,
                                         layout=control_layout)
        self.display_panel = widgets.Box(self.controls_display,
                                         layout=control_layout)
        self.csulc_panel = widgets.VBox(
            [widgets.HBox([self.all_csulc_button, self.none_csulc_button],
                          layout={'width': sidepanel_width}),
             widgets.VBox(self.controls_csulc,
                          layout=csulc_layout)],
            layout=control_layout)
        self.control_panel = widgets.Tab(children=[self.select_panel,
                                                   self.display_panel,
                                                   self.csulc_panel])
        self.control_panel.set_title(0, 'Selection')
        self.control_panel.set_title(1, 'Display')
        self.control_panel.set_title(2, 'Labels')
        # Copy over the start/default values.
        sid = self.sid_select.value
        hemi = self.hemi_select.value.lower()
        # Next, we setup the figure.
        subdata = subject_data[(sid, hemi)]
        segs = subdata['wang']
        im0 = plot_imcat(subdata, grid, start_cd['image'])
        imshape = im0.shape[:2]
        self.imshape = imshape
        (im_rs, im_cs) = imshape
        figw_px = figsize * dpi
        figh_px = figw_px * im_rs // im_cs
        figshape = (figh_px, figw_px)
        (figh, figw) = [('%dpx' % q) for q in figshape]
        (dot_rs, dot_cs) = (im_rs*grid_rs, im_cs*grid_cs)
        (fig,ax) = plt.subplots(
            constrained_layout=True,
            figsize=(figsize, figsize*dot_rs/dot_cs),
            dpi=dpi)
        self.figure = fig
        self.axes = ax
        fig.canvas.toolbar_visible = False
        fig.canvas.title_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        #ax.format_coord = lambda x,y: ''
        # Make the legend axes
        self.legend_axes = fig.add_axes([0.4,0.4,0.2,0.2])
        legim = legend_data[hemi].get(self.start_contour, None)
        if legim is None:
            legim = np.zeros((10,10,4))
        self.legend_implot = self.legend_axes.imshow(legim)
        self.legend_axes.axis('equal')
        self.legend_axes.axis('off')
        # Draw wang and set it's initial visibility.
        self.wang_plot = segs_decorate_plot(
            ax, segs, color=self.wang_color.value, lw=0.3, zorder=10,
            grid=grid, imshape=imshape)
        for ln in self.wang_plot:
            ln.set_visible(self.wang_shown.value)
        # Draw V123 and set their initial visibility.
        segs = subdata['v123']
        self.v123_plot = segs_decorate_plot(
            ax, segs, color=self.v123_color.value, lw=0.3, zorder=9,
            grid=grid, imshape=imshape)
        for ln in self.wang_plot:
            ln.set_visible(self.wang_shown.value)
        # Draw the CSulc contours and their initial visibilities.
        self.csulc_plots = {
            k: segs_decorate_plot(
                ax, {k:segs}, color=self.csulc_colors[k].value, lw=0.33, zorder=8,
                grid=grid, imshape=imshape)
            for (k,segs) in subdata['csulc'].items()}
        for (k,lns) in self.csulc_plots.items():
            for ln in lns:
                ln.set_visible(self.csulc_showns[k].value)

        # Initialize the display for this subject/hemi
        self.image_plot = ax.imshow(im0)
        ax.axis('off')
        # Setup all the listener functions...
        self.sid_select.observe(curry(self.update, 'sid'), 'value')
        self.hemi_select.observe(curry(self.update, 'hemi'), 'value')
        self.contour_select.observe(curry(self.update, 'contour'), 'value')
        self.wang_shown.observe(curry(self.update, 'wang'), 'value')
        self.work_shown.observe(curry(self.update, 'work'), 'value')
        self.v123_shown.observe(curry(self.update, 'v123'), 'value')
        self.wang_color.observe(curry(self.update, 'wang_color'), 'value')
        self.draw_color.observe(curry(self.update, 'draw_color'), 'value')
        self.work_color.observe(curry(self.update, 'work_color'), 'value')
        self.v123_color.observe(curry(self.update, 'v123_color'), 'value')
        for (_,_,a,_) in csulc_labels:
            self.csulc_showns[a].observe(curry(self.update, a),
                                         'value')
            self.csulc_colors[a].observe(curry(self.update, f'{a}_color'),
                                         'value')
        self.all_csulc_button.on_click(lambda b:self.csulc_all())
        self.none_csulc_button.on_click(lambda b:self.csulc_none())
        self.notes_area.observe(curry(self.update, 'notes'), 'value')
        self.save_button.on_click(lambda b:self.save())
        self.reset_button.on_click(lambda b:self.reset())
        self.canvas_conns = [
            #fig.canvas.mpl_connect('close_event', self.on_close),
            fig.canvas.mpl_connect('button_press_event', self.on_click)]
        # Final touches:
        self.work_plot = []
        self.draw_work()
        self.draw_plot = []
        self.redraw_contour()
        self.notes = None
        self.load_notes()
        self.outer_panel = widgets.HBox(
            [self.control_panel, fig.canvas],
            layout=widgets.Layout(
                flex_flow='row',
                align_items='center',
                width='100%',
                height=('%dpx' % (figh_px+6)),
                border='#000000'))
        display(self.outer_panel)
        # For saving errors that get caught in events:
        self._event_error = None

    # Basic accessors for the current settings:
    def curr_sid(self, newval=None):
        return int(self.sid_select.value)
    def curr_hemi(self):
        return self.hemi_select.value.lower()
    def curr_subdata(self):
        return subject_data.get((self.curr_sid(), self.curr_hemi()))
    def curr_contour(self):
        return self.contour_select.value
    def curr_work_shown(self):
        return self.work_shown.value
    def curr_wang_shown(self):
        return self.wang_shown.value
    def curr_v123_shown(self):
        return self.v123_shown.value
    def curr_draw_color(self):
        return self.draw_color.value
    def curr_work_color(self):
        return self.work_color.value
    def curr_wang_color(self):
        return self.wang_color.value
    def curr_v123_color(self):
        return self.v123_color.value

    # Methods that update all the various drawings.
    def update_image(self):
        subdata = self.curr_subdata()
        contour = self.curr_contour()
        cdat = contours[contour]
        im0 = plot_imcat(subdata, self.grid, cdat['image'])
        self.image_plot.set_data(im0)
    def update_lines(self, segs, old_plots, vis, color, lw=0.3, zorder=10):
        # Remove the old lines first, if need-be.
        for ln in old_plots:
            ln.remove()
        # Update the plot object.
        new_plots = segs_decorate_plot(self.axes, segs,
                                       grid=self.grid,
                                       imshape=self.imshape,
                                       color=color,
                                       lw=lw,
                                       zorder=zorder)
        # Make sure the visibility is correct.
        for ln in new_plots:
            ln.set_visible(vis)
        return new_plots
    def update_wang(self):
        old_plots = self.wang_plot
        vis = self.curr_wang_shown()
        color = self.curr_wang_color()
        lw = 0.3
        zorder = 10
        segs = self.curr_subdata()['wang']
        plots = self.update_lines(segs, old_plots, vis, color, lw, zorder)
        self.wang_plot = plots
    def update_v123(self):
        old_plots = self.v123_plot
        vis = self.curr_v123_shown()
        color = self.curr_v123_color()
        lw = self.contour_lw
        zorder = 11
        segs = self.curr_subdata()['v123']
        plots = self.update_lines(segs, old_plots, vis, color, lw, zorder)
        self.v123_plot = plots
    def update_csulc(self):
        olds = self.csulc_plots
        vis = {k:u.value for (k,u) in self.csulc_showns.items()}
        clr = {k:u.value for (k,u) in self.csulc_colors.items()}
        lw = self.csulc_lw
        zorder = 12
        segs = self.curr_subdata()['csulc']
        plots = {k: self.update_lines({k:v}, olds[k], vis[k], clr[k], lw, zorder)
                 for (k,v) in segs.items()}
        self.csulc_plots = plots
    def update_selection(self, sid=None, hemi=None, contour=None, save=True):
        if sid is None:
            sid = self.curr_sid()
        else:
            self.sid_select.value = sid
            self.contour_select.value = self.start_contour
            self.work_shown.value = True
            redraw_wang = True
        if hemi is None:
            hemi = self.curr_hemi()
        else:
            self.hemi_select.value = hemi.upper()
            self.contour_select.value = self.start_contour
            self.work_shown.value = True
            redraw_wang = True
        if contour is None:
            contour = self.curr_contour()
        else:
            self.contour_select.value = contour
            redraw_wang = False
        if save:
            self.save()
        # What's the new control selection:
        subdata = subject_data[(sid, hemi)]
        # Update the decor, the work, and the current drawings.
        if redraw_wang: self.update_wang()
        self.update_image()
        self.update_v123()
        self.update_csulc()
        self.draw_work()
        self.redraw_contour()
        # Redraw the legend.
        self.redraw_legend()
        # Update the notes
        self.notes_area.value = self.notes[sid][hemi][contour][0]
    def update(self, var, change):
        # What updated?
        if var == 'sid':
            # What's the new control selection:
            sid = int(change.new)
            # Run the update!
            self.update_selection(sid=sid)
        elif var == 'hemi':
            h = change.new.lower()
            self.update_selection(hemi=h)
        elif var == 'contour':
            contour = change.new
            self.update_selection(contour=contour)
        elif var == 'v123':
            wang = change.new
            for ln in self.v123_plot: ln.set_visible(wang)
        elif var == 'wang':
            wang = change.new
            for ln in self.wang_plot: ln.set_visible(wang)
        elif var == 'work':
            c = change.new
            for ln in self.work_plot: ln.set_visible(c)
        elif var == 'v123_color':
            c = change.new
            for ln in self.v123_plot: ln.set_color(c)
        elif var == 'wang_color':
            c = change.new
            for ln in self.wang_plot: ln.set_color(c)
        elif var == 'work_color':
            c = change.new
            for ln in self.work_plot: ln.set_color(c)
        elif var == 'draw_color':
            c = change.new
            for ln in self.draw_plot: ln.set_color(c)
        elif var.endswith('_color'):
            c = change.new
            for ln in self.csulc_plots[var[:-6]]: ln.set_color(c)
        elif var in csulc_colors:
            c = change.new
            for ln in self.csulc_plots[var]: ln.set_visible(c)
        elif var == 'notes':
            sid = self.curr_sid()
            h = self.curr_hemi()
            contour = self.curr_contour()
            self.notes[sid][h][contour][0] = change.new
            # no need to redraw
            return None
        else: return None
        self.figure.canvas.draw_idle()
        return None
    
    # Setup the figure clicks!
    def on_click(self, event):
        try:
            ax = self.axes
            fig = self.figure
            if event.inaxes != ax: return
            sid = self.curr_sid()
            h = self.curr_hemi()
            contour = self.curr_contour()
            cplot = self.draw_plot
            # if shift is down, we delete the last point
            ctrlkeys = ['control', 'ctrl']
            bothkeys = ['shift+control', 'shift+ctrl', 'control+shift', 'ctrl+shift']
            if event.key in ctrlkeys: # control means delete
                self.rmlast_click()
            elif event.key in bothkeys:
                self.rmfirst_click()
            elif event.key == 'shift': # shift means front instead of end
                self.prepend_click((event.xdata, event.ydata))
            else: # add the points
                self.append_click((event.xdata, event.ydata))
            fig.canvas.draw()
        except Exception as e:
            self._event_error = sys.exc_info()
            raise
    def draw_work(self):
        for ln in self.work_plot:
            ln.remove()
        sid = self.curr_sid()
        h = self.curr_hemi()
        subdata = subject_data[(sid,h)]
        contour = self.curr_contour()
        color = self.curr_work_color()
        vis = self.curr_work_shown()
        ax = self.axes
        plots = []
        for c in contours.keys():
            if c == contour: continue
            pts = self.clicks[sid][h][c]
            plots += clicks_decorate_plot(
                ax, pts, '.--',
                grid=self.grid, imshape=self.imshape,
                color=color,
                lw=self.contour_lw*0.75, ms=self.contour_ms/4)
        for p in plots:
            p.set_visible(vis)
        self.work_plot = plots
    def _get_subdir(self, sid):
        flnm = os.path.join(self.savedir, str(sid))
        if os.path.isdir(self.savedir) and not os.path.isdir(flnm):
            os.makedirs(flnm, mode=0o755)
        return flnm
    def load_clicks(self):
        def load_click_file(sid,h,c,subdir):
            flnm = os.path.join(subdir, f'{h}.{c}.json')
            if os.path.isfile(flnm):
                return ny.load(flnm)
            else:
                return []
        cl = {}
        for sid in subject_list:
            subdir = self._get_subdir(sid)
            r = {}
            for h in ['lh','rh']:
                rr = {}
                for contour in contours.keys():
                    c = contours[contour]['save']
                    rr[contour] = ny.util.curry(load_click_file,
                                                sid, h, c, subdir)
                r[h] = pimms.lmap(rr)
            cl[sid] = r
        self.clicks = cl
        self.clicks_updated = {}
    def csulc_all(self):
        for plots in self.csulc_plots.values():
            for ln in plots:
                ln.set_visible(True)
        for box in self.csulc_showns.values():
            box.value = True
    def csulc_none(self):
        for plots in self.csulc_plots.values():
            for ln in plots:
                ln.set_visible(False)
        for box in self.csulc_showns.values():
            box.value = False
    def save(self):
        self.save_clicks()
        self.save_notes()
    def reset(self):
        self.reset_clicks()
        self.reset_notes()
    def save_clicks(self):
        for ((sid,h,contour),orig) in self.clicks_updated.items():
            subdir = self._get_subdir(sid)
            c = contours[contour]['save']
            flnm = os.path.join(subdir, f'{h}.{c}.json')
            ny.save(flnm, self.clicks[sid][h][contour])
        # At this point, the clicks are no longer "updated"
        self.clicks_updated = {}
    def reset_clicks(self):
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        tup = (sid,h,contour)
        orig = self.clicks_updated.get(tup, None)
        newl = self.clicks[sid][h][contour]
        if orig is None: return None
        tmp = newl.copy()
        # Restore the originals:
        newl.clear()
        for el in orig:
            newl.append(el)
        self.redraw_contour()
        return tmp
    def redraw_contour(self):
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        cd = contours[contour]
        ax = self.axes
        pts = self.clicks[sid][h][contour]
        if len(pts) == 0 and 'start' in cd:
            # These are pinned to HCP lines
            (side,ln) = cd['start']
            ln = np.transpose(subject_data[(sid,h)]['v123'][ln])
            if side == 'start': pts = ln[[0]]
            elif side == 'end': pts = ln[[-1]]
            else:
                raise ValueError("start tuple must start with 'start' or 'end'")
            pts = flatmap_to_imgrid(pts)[0][0].T
            self.clicks[sid][h][contour].append(pts[0])
        for c in self.draw_plot:
            c.remove()
        self.draw_plot = []
        if len(pts) > 0:
            self.draw_plot = clicks_decorate_plot(
                ax, pts, 'o-',
                grid=self.grid,
                imshape=self.imshape,
                color=self.curr_draw_color(),
                lw=self.contour_lw,
                ms=self.contour_ms)
        return None
    def append_click(self, pt):
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        cl0.append(pt)
        self.redraw_contour()
        return None
    def prepend_click(self, pt):
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        cd = contours[contour]
        if 'start' in cd: return None
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        cl0.insert(0, pt)
        self.redraw_contour()
        return None
    def rmlast_click(self):
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        if len(cl0) == 0: return None
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        cl0.pop()
        self.redraw_contour()
        return None
    def rmfirst_click(self):
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        cd = contours[contour]
        if 'start' in cd: return None
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        if len(cl0) == 0: return None
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        del cl0[0]
        self.redraw_contour()
        return None
    def load_notes(self):
        def load_notes_file(sid,h,c,subdir):
            flnm = os.path.join(subdir, f'{h}.{c}_notes.txt')
            if os.path.isfile(flnm):
                s = ny.load(flnm)
                return [s, s]
            else:
                return ['', '']
        notes = {}
        for sid in subject_list:
            subdir = self._get_subdir(sid)
            r = {}
            for h in ['lh','rh']:
                rr = {}
                for contour in contours.keys():
                    c = contours[contour]['save']
                    rr[contour] = ny.util.curry(load_notes_file,
                                                sid, h, c, subdir)
                r[h] = pimms.lmap(rr)
            notes[sid] = r
        self.notes = notes
        # update the notes if need-be
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        self.notes_area.value = notes[sid][h][contour][0]
        return None
    def save_notes(self):
        for (sid,uu) in self.notes.items():
            for (h,u) in uu.items():
                for contour in u.keys():
                    if u.is_lazy(contour): continue
                    v = u[contour]
                    if v[0] == v[1]: continue
                    c = contours[contour]['save']
                    subdir = self._get_subdir(sid)
                    flnm = os.path.join(subdir, f'{h}.{c}_notes.txt')
                    ny.save(flnm, v[0])
                    v[1] = v[0]
        return None
    def reset_notes(self):
        for (sid,uu) in self.notes.items():
            for (h,u) in uu.items():
                for c in u.keys():
                    if u.is_lazy(c): continue
                    v = u[c]
                    if v[0] == v[1]: continue
                    v[0] = v[1]
        # reset the notes area:
        sid = self.curr_sid()
        h = self.curr_hemi()
        contour = self.curr_contour()
        self.notes_area.value = self.notes[sid][h][contour][0]
    def redraw_legend(self):
        hemi = self.curr_hemi()
        contour = self.curr_contour()
        legim = legend_data[hemi].get(contour, None)
        if legim is None:
            legim = np.zeros((10,10,4))
        self.legend_implot.set_data(legim)

        
        
        
        
