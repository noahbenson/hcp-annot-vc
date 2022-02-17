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

# The default image size; this is the assumed size of the images that are
# downloaded from the OSF and displayed, in pixels.
default_imshape = (864*2, 864*2)
# The default x- and y-value limits of the flatmaps that were used for
# generating the images.
default_xlim = (-100, 100)
default_ylim = (-100, 100)
# The default grid used for the display. The None is a stand-in for the image of
# the current contour's highlight.
default_grid = ((None,        'polar_angle'),
                ('curvature', 'eccentricity'))
# The path that we load images from by default.
default_load_path = '/data'
default_osf_url = 'osf://tery8/'

# The HCP Retinotopy subjects:
subject_ids = (100610, 102311, 102816, 104416, 105923, 108323, 109123, 111312,
               111514, 114823, 115017, 115825, 116726, 118225, 125525, 126426,
               128935, 130114, 130518, 131217, 131722, 132118, 134627, 134829,
               135124, 137128, 140117, 144226, 145834, 146129, 146432, 146735,
               146937, 148133, 150423, 155938, 156334, 157336, 158035, 158136,
               159239, 162935, 164131, 164636, 165436, 167036, 167440, 169040,
               169343, 169444, 169747, 171633, 172130, 173334, 175237, 176542,
               177140, 177645, 177746, 178142, 178243, 178647, 180533, 181232,
               181636, 182436, 182739, 185442, 186949, 187345, 191033, 191336,
               191841, 192439, 192641, 193845, 195041, 196144, 197348, 198653,
               199655, 200210, 200311, 200614, 201515, 203418, 204521, 205220,
               209228, 212419, 214019, 214524, 221319, 233326, 239136, 246133,
               249947, 251833, 257845, 263436, 283543, 318637, 320826, 330324,
               346137, 352738, 360030, 365343, 380036, 381038, 385046, 389357,
               393247, 395756, 397760, 401422, 406836, 412528, 429040, 436845,
               463040, 467351, 525541, 536647, 541943, 547046, 550439, 552241,
               562345, 572045, 573249, 581450, 585256, 601127, 617748, 627549,
               638049, 644246, 654552, 671855, 680957, 690152, 706040, 724446,
               725751, 732243, 751550, 757764, 765864, 770352, 771354, 782561,
               783462, 789373, 814649, 818859, 825048, 826353, 833249, 859671,
               861456, 871762, 872764, 878776, 878877, 898176, 899885, 901139,
               901442, 905147, 910241, 926862, 927359, 942658, 943862, 951457,
               958976, 966975, 971160, 973770, 995174)

def imgrid_to_flatmap(pts,
                      grid=default_grid,
                      imshape=default_imshape,
                      xlim=default_xlim,
                      ylim=default_ylim):
    '''
    `imgrid_to_flatmap(pts)` yields a 2xN matrix the same size as the given
      (2xN) matrix `pts`, for which the points have been converted from
      coordinates in the given image grid (`grid` option).
    '''
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
    rpx2yu = -(ymax - ymin) / r
    cpx2xu = (xmax - xmin) / c
    (c,r) = pts if pts.shape[0] == 2 else pts.T
    while True:
        ii = c > cs
        if len(ii) == 0: break
        c[ii] -= cs
    while True:
        ii = r > rs
        if len(ii) == 0: break
        r[ii] -= rs
    x = xmu + (cs - cmu)*cpx2xu
    y = ymu + (rs - rmu)*rpx2yu
    return np.array([x,y])
def flatmap_to_imgrid(pts,
                      grid=default_grid,
                      imshape=default_imshape,
                      xlim=default_xlim,
                      ylim=default_ylim):
    '''
    `flatmap_to_imgrid(pts)` yields a 2xN matrix the same size as the given
      (2xN) matrix `pts`, for which the points have been converted from
      coordinates in the default flatmap representation to the given
      image grid (`grid` option).
    '''
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
def clicks_update_plot(ax, plots, pts, grid=default_grid, imshape=default_imshape):
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

# Functions for loading data.
def load_sub_v123(sid):
    path = ny.util.pseudo_path(default_osf_url,
                               cache_path=default_load_path)
    path = path.local_path('annot-v123', '%d.json.gz' % (sid,))
    return ny.load(path)
def load_subimage(sid, h, name,
                  load_path=default_load_path, osf_url=default_osf_url):
    from PIL import Image
    flnm = os.path.join(load_path, str(sid), '%d_%s_%s.png' % (sid, h, name))
    with Image.open(flnm) as im:
        arr = np.array(im)
    return arr
def curry_load_subimage(sid, h, name,
                        load_path=default_load_path, osf_url=default_osf_url):
    return lambda:load_subimage(sid, h, name,
                                load_path=load_path, osf_url=osf_url)
def load_subwang(sid, h, load_path=default_load_path, osf_url=default_osf_url):
    import neuropythy as ny
    flnm = os.path.join(load_path, str(sid), '%d_%s_wang.mgz' % (sid, h))
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
                            for s in subject_ids})
def prep_subdata(sid, h, load_path=None, osf_url=default_osf_url):
    if load_path is None: load_path = default_load_path
    if osf_url is None: osf_url = default_osf_url
    dirname = os.path.join(load_path, str(sid))
    if not os.path.isfile(dirname):
        pp = ny.util.pseudo_path(osf_url)
        path = pp.local_path('annot-images', '%d.tar.gz' % sid)
        import tarfile
        with tarfile.open(path) as fl:
            fl.extractall(load_path)
    ims = {imname: curry_load_subimage(sid, h, imname,
                                       load_path=load_path, osf_url=osf_url)
           for imname in image_order}
    ims['wang'] = lambda:load_subwang(sid, h,
                                      load_path=load_path, osf_url=osf_url)
    ims['v123'] = lambda:v123_contours[sid][h]
    return pimms.lmap(ims)
def curry_prep_subdata(sid, h,
                       load_path=default_load_path, osf_url=default_osf_url):
    return lambda:prep_subdata(sid, h, load_path=load_path, osf_url=osf_url)

subject_data = pimms.lmap({(sid,h): curry_prep_subdata(sid, h)
                           for sid in subject_ids
                           for h in ['lh','rh']})

contour_data = [
    # hV4:
    dict(name='hV4/VO1 Boundary', save='hV4_VO1', legend='hV4_VO1',
         image='eccpeak_6.25deg'),
    dict(name='hV4/VO1 Middle (*)', save='hV4_VO1_mid', legend='isoang_hV4_VO1_mid',
         image='isoang_90', optional=True),
    dict(name='hV4 Ventral Boundary', save='hV4', legend='hV4_ventral',
         image='isoang_vml', start=('end', 'V3_ventral')),
    dict(name='VO1+VO2 Outer Boundary', save='VO_outer', legend='VO_outer',
         image='isoang_vml', start=('start', 'V3_ventral')),
    dict(name='VO1/VO2 Interior Boundary', save='VO1_VO2', legend='VO1_VO2',
         image='isoang_vmu'),
    # V3A/B
    #dict(name='V3A/B Outer Boundary', save='V3ab_outer', legend='V3ab_outer',
    #     image='V3ab_outer'),
    #dict(name='V3A/B Inner Boundary', save='V3ab_inner', legend='V3ab_inner',
    #     image='V3ab_inner'),
]
contours = {cd['name']: cd for cd in contour_data}
default_start_contour = contour_data[0]['name']

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
def prep_legends(load_path=default_load_path, osf_url=default_osf_url):
    dirname = os.path.join(load_path, 'legends')
    if not os.path.isfile(dirname):
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
                 start_contour=default_start_contour,
                 grid=default_grid, dpi=72*8,
                 contour_lw=0.25, contour_ms=0.25):
        # Copy over the simple parameters of the class first.
        self.grid = grid
        self.start_contour = start_contour
        self.contour_lw = contour_lw
        self.contour_ms = contour_ms
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
            options=subject_ids,
            value=subject_ids[0],
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
        self.save_button = widgets.Button(description='Save.')
        self.reset_button = widgets.Button(description='Reset.')
        self.save_box = widgets.HBox(
            children=[self.save_button, self.reset_button],
            layout={'align_items': 'center'})
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
        self.controls = self.controls_select + self.controls_display
        # Go ahead and make the control panel for both the selection and the
        # display tabs.
        control_layout = {'height': f"{figh*dpi*0.65}px",
                          'width': sidepanel_width,
                          'display': 'flex',
                          'flex_flow': 'column',
                          'flex_wrap': 'nowrap',
                          'align_items': 'center',
                          'justify_content': 'flex-start'}
        self.select_panel = widgets.Box(self.controls_select,
                                         layout=control_layout)
        self.display_panel = widgets.Box(self.controls_display,
                                         layout=control_layout)
        self.control_panel = widgets.Tab(children=[self.select_panel, self.display_panel])
        self.control_panel.set_title(0, 'Selection')
        self.control_panel.set_title(1, 'Display')
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
        self.legend_axes = fig.add_axes([0.35,0.35,0.3,0.3])
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
        # Initialize the display for this subject/hemi
        self.image_plot = ax.imshow(im0)
        ax.axis('off')
        # Setup all the listener functions...
        self.sid_select.observe(ny.util.curry(self.update, 'sid'), 'value')
        self.hemi_select.observe(ny.util.curry(self.update, 'hemi'), 'value')
        self.contour_select.observe(ny.util.curry(self.update, 'contour'), 'value')
        self.wang_shown.observe(ny.util.curry(self.update, 'wang'), 'value')
        self.work_shown.observe(ny.util.curry(self.update, 'work'), 'value')
        self.v123_shown.observe(ny.util.curry(self.update, 'v123'), 'value')
        self.wang_color.observe(ny.util.curry(self.update, 'wang_color'), 'value')
        self.draw_color.observe(ny.util.curry(self.update, 'draw_color'), 'value')
        self.work_color.observe(ny.util.curry(self.update, 'work_color'), 'value')
        self.v123_color.observe(ny.util.curry(self.update, 'v123_color'), 'value')
        self.notes_area.observe(ny.util.curry(self.update, 'notes'), 'value')
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
        for sid in subject_ids:
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
        for sid in subject_ids:
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

        
        
        
        
