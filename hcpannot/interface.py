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
default_pseudo_path = ny.util.pseudo_path(default_osf_url,
                                          cache_path=default_load_path)

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
    pts_nx2 = np.reshape(segs, (-1, 2))
    rcs = flatmap_to_imgrid(pts_nx2.T, grid=grid, imshape=imshape,
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
    path = default_pseudo_path.local_path('annot-v123', '%d.json.gz' % (sid,))
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
    grid = [[ims[k if g is None else g] for g in row]
            for row in grid]
    return imcat(grid)
# We can (lazily) load the V1-V3 contours now (we could altrnately load them in
# prep_subdata() function, but this prevents them from being loaded once for
# each hemisphere).
v123_contours = pimms.lmap({s: ny.util.curry(load_sub_v123, s)
                            for s in subject_ids})
def prep_subdata(sid, h, load_path=default_load_path, osf_url=default_osf_url):
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

boundary_contours = {'hV4 Middle':  'isoang_90',
                     'hV4/Outer Boundary': 'isoang_vml'}
contour_names = tuple(list(boundary_contours.keys()) + 
                      ['Ventral 0° iso-eccen', 'Ventral 0.5° iso-eccen'] + 
                      ['Vental %d° iso-eccen' % k for k in [1,2,4,7]] +
                      ['Dorsal 0° iso-eccen',  'Dorsal 0.5° iso-eccen'] + 
                      ['Dorsal %d° iso-eccen' % k for k in [1,2,4,7]])
contour_key = dict(boundary_contours)
contour_key['Ventral 0.5° iso-eccen'] = 'isoecc_0.5'
contour_key['Dorsal 0.5° iso-eccen'] = 'isoecc_0.5'
for k in [0,1,2,4,7]:
    contour_key['Ventral %d° iso-eccen' % k] = 'isoecc_%d' % k
    contour_key['Dorsal %d° iso-eccen' % k] = 'isoecc_%d' % k
contour_key = pyr.pmap(contour_key)
contour_save_key = pyr.pmap(
    {'hV4 Middle': 'isoang_hV4m',
     'hV4/Outer Boundary': 'isoang_hV4v',
     'Ventral 0° iso-eccen': 'isoecc_0v',
     'Ventral 0.5° iso-eccen': 'isoecc_0pt5v',
     'Ventral 1° iso-eccen': 'isoecc_1v',
     'Ventral 2° iso-eccen': 'isoecc_2v',
     'Ventral 4° iso-eccen': 'isoecc_4v',
     'Ventral 7° iso-eccen': 'isoecc_7v',
     'Dorsal 0° iso-eccen': 'isoecc_0d',
     'Dorsal 0.5° iso-eccen': 'isoecc_0pt5d',
     'Dorsal 1° iso-eccen': 'isoecc_1d',
     'Dorsal 2° iso-eccen': 'isoecc_2d',
     'Dorsal 4° iso-eccen': 'isoecc_4d',
     'Dorsal 7° iso-eccen': 'isoecc_7d'})
legend_key = {'hV4_mid':     'hV4 Middle',
              'hV4_ventral': 'hV4/Outer Boundary',
              '0v':          'Ventral 0° iso-eccen',
              '0.5v':        'Ventral 0.5° iso-eccen',
              '1v':          'Ventral 1° iso-eccen',
              '2v':          'Ventral 2° iso-eccen',
              '4v':          'Ventral 4° iso-eccen',
              '7v':          'Ventral 7° iso-eccen',
              '0d':          'Dorsal 0° iso-eccen',
              '0.5d':        'Dorsal 0.5° iso-eccen',
              '1d':          'Dorsal 1° iso-eccen',
              '2d':          'Dorsal 2° iso-eccen',
              '4d':          'Dorsal 4° iso-eccen',
              '7d':          'Dorsal 7° iso-eccen'}
legend_rkey = {v:k for (k,v) in legend_key.items()}

def load_legimage(load_path, h, imname):
    from PIL import Image
    flname = legend_rkey[imname]
    flnm = os.path.join(load_path, 'legends', f'{h}_{flname}.png')
    with Image.open(flnm) as im:
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
    ims = {h: pimms.lmap({imname: curry_load_legimage(load_path, h, imname)
                          for imname in legend_key.values()})
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
                 start_contour='V3/Outer ventral',
                 grid=default_grid, dpi=72*8,
                 contour_lw=0.25, contour_ms=0.25):
        self.grid = grid
        self.start_contour = start_contour
        self.contour_lw = contour_lw
        self.contour_ms = contour_ms
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
        start_contour = contour_key[start_contour]
        (grid_rs, grid_cs) = (len(grid), len(grid[0]))
        figh = figsize * grid_rs / grid_cs
        # Go ahead and setup the Widgets.
        self.sid_select = widgets.Dropdown(
            options=subject_ids,
            value=subject_ids[0],
            description='SID:',
            layout={'width': dropdown_width})
        self.hemi_select = widgets.Dropdown(
            options=['LH','RH'],
            value='LH',
            description='Hemi:',
            layout={'width': dropdown_width})
        self.line_select = widgets.Dropdown(
            options=contour_names,
            value=self.start_contour,
            description='Contour:',
            layout={'width': dropdown_width})
        self.anat_shown = widgets.Checkbox(
            description='Wang et al. (2015) Contours',
            value=False)
        self.anat_color = widgets.ColorPicker(
            description='Wang Color:',
            concise=True,
            value='white',
            layout={'width':'50%'})
        self.contour_shown = widgets.Checkbox(
            description='Drawn Contour',
            value=True)
        self.contour_color = widgets.ColorPicker(
            description='Draw Color:',
            concise=True,
            value='cyan',
            layout={'width':'50%'})
        self.notes_area = widgets.Textarea(
            value='', 
            description='',
            layout={'width': '95%', 'height': sidepanel_width})
        self.notes_panel = widgets.VBox(
            [widgets.Label('Contour Notes:'), self.notes_area],
            layout={'align_items': 'flex-start', 'width':'100%'})
        self.save_button = widgets.Button(description='Save.')
        self.reset_button = widgets.Button(description='Reset.')
        #center_layout = widgets.Layout(align_items='center')
        self.save_box = widgets.HBox(
            children=[self.save_button, self.reset_button],
            layout={'align_items': 'center'})
        self.controls = (self.sid_select,
                         self.hemi_select,
                         self.line_select,
                         self.anat_shown,
                         self.anat_color,
                         self.contour_shown,
                         self.contour_color,
                         self.notes_panel,
                         self.save_button,
                         self.reset_button)
        # The start/default values:
        self.sid = self.sid_select.value
        self.hemi = self.hemi_select.value.lower()
        # Setup the figure.
        subdata = subject_data[(self.sid, self.hemi)]
        segs = subdata['wang']
        im0 = plot_imcat(subdata, grid, start_contour)
        imshape = im0.shape[:2]
        self.imshape = imshape
        (im_rs, im_cs) = imshape
        figw_px = figsize * dpi
        figh_px = figw_px * im_rs // im_cs
        figshape = (figh_px, figw_px)
        (figh, figw) = [('%dpx' % q) for q in figshape]
        self.control_panel = widgets.VBox(self.controls, layout={'height':'100%'})
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
        legim = legend_data[self.hemi][self.start_contour]
        self.legend_implot = self.legend_axes.imshow(legim)
        self.legend_axes.axis('equal')
        self.legend_axes.axis('off')
        self.wang_plot = segs_decorate_plot(
            ax, segs, color=self.anat_color.value, lw=0.3, zorder=10,
            grid=grid, imshape=imshape)
        for ln in self.wang_plot:
            ln.set_visible(self.anat_shown.value)
        # Initialize the display for this subject/hemi
        self.image_plot = ax.imshow(im0)
        ax.axis('off')
        # Setup the listener functions...
        self.sid_select.observe(ny.util.curry(self.update, 'sid'), 'value')
        self.hemi_select.observe(ny.util.curry(self.update, 'hemi'), 'value')
        self.line_select.observe(ny.util.curry(self.update, 'line'), 'value')
        self.anat_shown.observe(ny.util.curry(self.update, 'anat'), 'value')
        self.contour_shown.observe(ny.util.curry(self.update, 'contour'), 'value')
        self.anat_color.observe(ny.util.curry(self.update, 'anatcolor'), 'value')
        self.contour_color.observe(ny.util.curry(self.update, 'contourcolor'), 'value')
        self.notes_area.observe(ny.util.curry(self.update, 'notes'), 'value')
        self.save_button.on_click(lambda b:self.save())
        self.reset_button.on_click(lambda b:self.reset())
        self.canvas_conns = [
            #fig.canvas.mpl_connect('close_event', self.on_close),
            fig.canvas.mpl_connect('button_press_event', self.on_click)]
        # Final touches:
        self.bg_contour_plot = []
        self.draw_bg_contours()
        self.contour_plot = []
        self.redraw_contours()
        self.notes = None
        self.load_notes()
        self.control_panel.layout = widgets.Layout(width=sidepanel_width,
                                                   height='100%',
                                                   align_items='center')
        pane = widgets.HBox(
            [self.control_panel, fig.canvas],
            layout=widgets.Layout(
                flex_flow='row',
                align_items='center',
                width='100%',
                height=('%dpx' % (figh_px+6)),
                border='#000000'))
        display(pane)
        # For saving errors that get caught in events:
        self._event_error = None

    def update(self, var, change):
        sid = int(self.sid_select.value)
        h = self.hemi_select.value.lower()
        contour = self.line_select.value
        anat = self.anat_shown.value
        ax = self.axes
        fig = self.figure
        # Get & remove the (now deprecated) plots:
        implot = self.image_plot
        wangplot = self.wang_plot
        # What updated?
        if var == 'sid':
            self.save()
            # What's the new control selection:
            sid = int(change.new)
            h = self.hemi.lower()
            # Remove contour plots if need-be
            for c in self.contour_plot:
                c.remove()
            self.contour_plot = []
            # New plots:
            subdata = subject_data[(sid, h)]
            contour = self.start_contour
            c = contour_key[contour]
            im0 = plot_imcat(subdata, self.grid, c)
            self.image_plot.set_data(im0)
            for ln in self.wang_plot: ln.remove()
            segs = subdata['wang']
            self.wang_plot = segs_decorate_plot(
                ax, segs,
                grid=self.grid, imshape=self.imshape,
                color=self.anat_color.value, lw=0.3, zorder=10)
            anat = self.anat_shown.value
            for ln in self.wang_plot: ln.set_visible(anat)
            pts = self.clicks[sid][h][contour]
            self.contour_plot = clicks_decorate_plot(
                ax, pts, 'o-',
                grid=self.grid, imshape=self.imshape,
                color=self.contour_color.value,
                lw=self.contour_lw, ms=self.contour_ms)
            # Update the output data:
            self.sid = sid
            self.draw_bg_contours()
            # Update the controls:
            #anat_shown.value = True
            self.line_select.value = self.start_contour
            self.contour_shown.value = True
            self.notes_area.value = self.notes[sid][h][contour][0]
            self.redraw_legend()
        elif var == 'hemi':
            self.save()
            # New plots:
            h = change.new.lower()
            subdata = subject_data[(self.sid, h)]
            contour = self.start_contour
            c = contour_key[contour]
            im0 = plot_imcat(subdata, self.grid, c)
            self.image_plot.set_data(im0)
            # Update Wang plot lines:
            for ln in self.wang_plot: ln.remove()
            segs = subdata['wang']
            self.wang_plot = segs_decorate_plot(
                ax, segs,
                grid=self.grid, imshape=self.imshape,
                color=self.anat_color.value, lw=0.3, zorder=10)
            anat = self.anat_shown.value
            for ln in self.wang_plot: ln.set_visible(anat)
            # And the drawn contours:
            for c in self.contour_plot: c.remove()
            pts = self.clicks[sid][h][contour]
            self.contour_plot = clicks_decorate_plot(
                ax, pts, 'o-',
                grid=self.grid, imshape=self.imshape,
                color=self.contour_color.value,
                lw=self.contour_lw, ms=self.contour_ms)
            # Update the output data:
            self.hemi = h
            self.draw_bg_contours()
            # Update the controls:
            #anat_shown.value = True
            self.line_select.value = self.start_contour
            self.contour_shown.value = True
            self.notes_area.value = self.notes[sid][h][contour][0]
            self.redraw_legend()
        elif var == 'line':
            self.save()
            # Remove contour plots if need-be
            for c in self.contour_plot: c.remove()
            contour = change.new
            c = contour_key[contour]
            subdata = subject_data[(sid,h)]
            im = plot_imcat(subdata, self.grid, c)
            self.image_plot.set_data(im)
            self.contour_shown.value = True
            # Redraw the chosen contours if need-be
            pts = self.clicks[sid][h][contour]
            self.contour_plot = clicks_decorate_plot(
                ax, pts, 'o-',
                grid=self.grid, imshape=self.imshape,
                color=self.contour_color.value,
                lw=self.contour_lw, ms=self.contour_ms)
            for ln in self.contour_plot:
                ln.set_visible(True)
            self.draw_bg_contours()
            self.notes_area.value = self.notes[sid][h][contour][0]
            self.redraw_legend()
        elif var == 'anat':
            anat = change.new
            for ln in self.wang_plot: ln.set_visible(anat)
        elif var == 'contour':
            c = change.new
            for ln in self.contour_plot: ln.set_visible(c)
            for ln in self.bg_contour_plot: ln.set_visible(c)
        elif var == 'anatcolor':
            c = change.new
            for ln in self.wang_plot: ln.set_color(c)
        elif var == 'contourcolor':
            c = change.new
            for ln in self.contour_plot: ln.set_color(c)
            for ln in self.bg_contour_plot: ln.set_color(c)
        elif var == 'notes':
            self.notes[sid][h][contour][0] = change.new
            # no need to redraw
            return None
        else: return None
        fig.canvas.draw_idle()
        return None
    
    # Setup the figure clicks!
    def on_click(self, event):
        if not self.contour_shown.value: return None
        try:
            ax = self.axes
            fig = self.figure
            if event.inaxes != ax: return
            sid = int(self.sid_select.value)
            h = self.hemi_select.value.lower()
            contour = self.line_select.value
            c = contour_key[contour]
            cplot = self.contour_plot
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
    def draw_bg_contours(self):
        for ln in self.bg_contour_plot: ln.remove()
        contour = self.line_select.value
        c = contour_key[contour]
        sid = self.sid
        h = self.hemi.lower()
        ax = self.axes
        subdata = subject_data[(sid,h)]
        plots = []
        for c in contour_names:
            if c == contour: continue
            pts = self.clicks[sid][h][c]
            plots += clicks_decorate_plot(
                ax, pts, '.:',
                grid=self.grid, imshape=self.imshape,
                color=self.contour_color.value,
                lw=self.contour_lw/2, ms=self.contour_ms/4)
        self.bg_contour_plot = plots
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
                for contour in contour_names:
                    c = contour_save_key[contour]
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
            c = contour_save_key[contour]
            flnm = os.path.join(subdir, f'{h}.{c}.json')
            ny.save(flnm, self.clicks[sid][h][contour])
        # At this point, the clicks are no longer "updated"
        self.clicks_updated = {}
    def reset_clicks(self):
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        tup = (sid,h,contour)
        orig = self.clicks_updated.get(tup, None)
        newl = self.clicks[sid][h][contour]
        if orig is None: return None
        tmp = newl.copy()
        newl = self.clicks[sid][h][contour]
        # Restore the originals:
        newl.clear()
        for el in orig:
            newl.append(el)
        self.redraw_contours()
        return tmp
    def redraw_contours(self):
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        ax = self.axes
        pts = self.clicks[sid][h][contour]
        for c in self.contour_plot: c.remove()
        self.contour_plot = []
        if len(pts) > 0:
            self.contour_plot = clicks_decorate_plot(
                ax, pts, 'o-',
                grid=self.grid, imshape=self.imshape,
                color=self.contour_color.value,
                lw=self.contour_lw, ms=self.contour_ms)
        return None
    def append_click(self, pt):
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        cl0.append(pt)
        self.redraw_contours()
        return None
    def prepend_click(self, pt):
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        cl0.insert(0, pt)
        self.redraw_contours()
        return None
    def rmlast_click(self):
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        if len(cl0) == 0: return None
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        cl0.pop()
        self.redraw_contours()
        return None
    def rmfirst_click(self):
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        tup = (sid,h,contour)
        cl0 = self.clicks[sid][h][contour]
        if len(cl0) == 0: return None
        orig = self.clicks_updated.get(tup, None)
        if orig is None:
            orig = cl0.copy()
            self.clicks_updated[tup] = orig
        del cl0[0]
        self.redraw_contours()
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
                for contour in contour_names:
                    c = contour_save_key[contour]
                    rr[contour] = ny.util.curry(load_notes_file,
                                                sid, h, c, subdir)
                r[h] = pimms.lmap(rr)
            notes[sid] = r
        self.notes = notes
        # update the notes if need-be
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        self.notes_area.value = notes[sid][h][contour][0]
        return None
    def save_notes(self):
        for (sid,uu) in self.notes.items():
            for (h,u) in uu.items():
                for contour in u.keys():
                    if u.is_lazy(contour): continue
                    v = u[contour]
                    if v[0] == v[1]: continue
                    c = contour_save_key[contour]
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
        sid = self.sid
        h = self.hemi.lower()
        contour = self.line_select.value
        self.notes_area.value = self.notes[sid][h][contour][0]
    def redraw_legend(self):
        contour = self.line_select.value
        legim = legend_data[self.hemi][contour]
        self.legend_implot.set_data(legim)

        
        
        
        
