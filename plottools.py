import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.transforms          import IdentityTransform, TransformedBbox, Bbox
from mpl_toolkits.axes_grid1        import make_axes_locatable
from vip_hci.preproc.recentering    import frame_center
from vip_hci.var.shapes             import get_square
from matplotlib.patches             import Arc


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])

def plot_angles(sub, x, y, index=0):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(sub.T, extent=[-sub.shape[1]/2., 
                            sub.shape[1]/2., 
                            -sub.shape[0]/2., 
                            sub.shape[0]/2. ])

    center = (0., 0.)
    p1 = [(0., 0.), (x, y)]
    p2 = [(0., 0.), (0., y)]
    
    axis_x = [(0., -1.2*y), (0., 1.2*y)]
    axis_y = [(-1.2*x, 0.), (1.2*x, 0.)]

    
    line1, = ax.plot(*zip(*p1))
    line2, = ax.plot(*zip(*p2))
    line3, = ax.plot(*zip(*axis_x), color='k')
    line4, = ax.plot(*zip(*axis_y), color='k')
    point, = ax.plot(*center, marker="o")

    am1 = AngleAnnotation(center, p1[1], p2[1], ax=ax, size=75, text=r"$\alpha$")
    # fig.savefig('./figures/negfc/{}.png'.format(index))
    plt.show()

def plot_to_compare(images, titles, axes=None, show=True, img_file=None, dpi=100, text_box='',**savefig_params):
    """ Plot a list of images and their corresponding titles
    
    :param images: A list of 2dim images
    :type images: list<numpy.ndarray>
    :param titles: A list of titles
    :type titles: list<string>
    :param axes: Matplotlib predefined axes, defaults to None
    :type axes: matplotlib.axes.Axes, optional
    :returns: Axes with the image plots
    :rtype: {matpllotlib.axes.Axes}
    """
    text_box = shape_text(text_box)
    if axes is None:
        fig, axes = plt.subplots(1, len(images), dpi=dpi,
            gridspec_kw={'hspace': 0.1, 'wspace': .5})
    for i, (im, ti) in enumerate(zip(images, titles)):
        im_obj = axes[i].imshow(im, origin = "lower")
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_obj, cax=cax)
        axes[i].set_title(ti)
    fig.text(0.2, 0.1, text_box, fontdict=None, backgroundcolor='thistle')
    if show:
        plt.show()
    if img_file:
        fig.savefig(img_file, **savefig_params)

    return axes 

def plot_detection(frame, table, bounded=True, dpi=100, text_box=''):
    ''' Plot a detection in a given frame
    
    By using a set of coordinates, this function draw a circle with center row[x], row[y]
    and radius == FWHM.
    :param frame: An 2-dimensional image from the cube=
    :type frame: np.ndarray
    :param table: Table containing candidate detections (needs `get_intersting_coords`)
    :type table: pandas.DataFrame
    :param bounded: If a zoom to the center is applied, defaults to True
    :type bounded: bool, optional
    '''
    text_box = shape_text(text_box)
    fig, ax = plt.subplots(1,1, dpi=dpi)
    ax.imshow(frame, origin="lower")
    fig.text(0., 0.1, text_box, fontdict=None, backgroundcolor='thistle')
    for _, row in table.iterrows():
        circle = plt.Circle((row['x'], row['y']), row['fwhm_mean'], fill=False, edgecolor='r')  
        ax.add_patch(circle)
    fig.tight_layout()
    plt.show()

def plot_cube(cube, save=False, root='./figures/cube_gif'):
    """ Plot each frame from a cube
    
    :param cube: A cube containing frames in a single wavelength
    :type cube: numpy.ndarray
    :param save: Write each frame figure, defaults to False
    :type save: bool, optional
    """
    os.makedirs(root, exist_ok=True)
    for i in range(cube.shape[0]):
        fig, axes = plt.subplots(1, 1, dpi=300, gridspec_kw={'hspace': 0., 'wspace': .4})       
        # y, x  = frame_center(cube[i])
        # frame = get_square(cube[i], size=40, y=y, x=x, position=False)
        im_obj = axes.imshow(cube[i], origin="lower")
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_obj, cax=cax)
        axes.set_title(r'$\lambda = H2$')
        fig.text(.38, .85, f'{i}-th frame from the cube', va='center', rotation='horizontal')
        if save:
            plt.savefig(f'{root}/{i}.png', format='png',  bbox_inches = "tight")
        else:
            plt.show()

def shape_text(text, wpr=10):
    final = ''
    for index, word in enumerate(text.split()):
        if index % wpr == 0 and index != 0:
            final += word+'\n'
        else:
            final += word+' '
    return final

def plot_cube_multiband(cube, save=False, dpi=100, root='./figures/cube_gif', text_box=''):
    """ Plot each frame from a cube
    
    :param cube: A cube containing frames
    :type cube: numpy.ndarray
    :param save: Write each frame figure, defaults to False
    :type save: bool, optional
    """
    text_box = shape_text(text_box)

    os.makedirs(root, exist_ok=True)
    n_bands = cube.shape[0]
    width = cube.shape[-1]
    fig, axes = plt.subplots(1, n_bands, dpi=dpi, gridspec_kw={'hspace': 0., 'wspace': .4})   
    for i in range(cube.shape[0]):
        for lambda_ in range(n_bands):
            # y, x  = frame_center(cube[i])
            # frame = get_square(cube[i], size=40, y=y, x=x, position=False)
            im_obj = axes[lambda_].imshow(cube[lambda_][i], origin="lower")
            divider = make_axes_locatable(axes[lambda_])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_obj, cax=cax)
            axes[lambda_].set_title(r'$\lambda = H{}$'.format(lambda_))
            fig.text(.38, .75, f'{i}-th frame from the cube', va='center', rotation='horizontal')

        fig.text(0.1, 0.1, text_box, fontdict=None, backgroundcolor='thistle')
        if save:
            plt.savefig(f'{root}/{i}.png', format='png',  bbox_inches = "tight")
        else:
            plt.show()
        break

def plot_optimization(frame, frame_negfc, msg='', root='./figures/negfc_opt/'):
    fig, axes = plt.subplots(1, 3, figsize=(5,5), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()
    axes[0].imshow(frame, origin="lower")
    axes[0].set_title('Frame')
    axes[1].imshow(frame_negfc, origin="lower")
    axes[1].set_title('Frame + FC')
    residuals = frame_negfc - frame
    axes[2].imshow(residuals, origin="lower")
    axes[2].set_title(msg)

    files = os.listdir(root)
    if len(files) == 0:
      fig.savefig(root+'0.png')
    else:
      numbers = [int(file.split('.png')[0]) for file in files]
      numbers = np.sort(numbers)
      fig.savefig(root+'{}.png'.format(numbers[-1]+1))

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def plot_mask(frame, posx, posy, fwhm):
    plt.figure(figsize=(5,5), dpi=300)
    h, w = frame.shape
    mask = create_circular_mask(h, w, center=(posx, posy), radius=fwhm)
    masked_frame = frame*mask
    plt.imshow(masked_frame, lower="origin")
    plt.scatter(posy, posx, marker='x', s=5, color='yellow')
    plt.plot([posy+fwhm, posy-fwhm], [posx, posx], linestyle='--', color='blue', linewidth=0.5)

    plt.show()

def plot_region(frame, posx, posy, xx=None, yy=None):
    plt.figure(figsize=(5,5), dpi=300)
    plt.imshow(frame, lower="origin")
    plt.scatter(posx, posy, marker='x', color='red')
    if xx is not None and yy is not None:
        plt.scatter(xx, yy, marker='x', color='yellow', s=0.1, alpha=0.5)
    plt.show()
