import numpy as np
import os

from vip_hci.fits import open_fits
from .preprocessing import modify_shape_and_center, parse_filter_code, shift_and_crop_cube, crop_cube
from .utils import load_dataset

def load_data(root, cropsize=None, n_jobs=1):

    if root.endswith('.pickle'):
        dataset = load_dataset(root)
        if cropsize is not None:
            dataset['cube'] = crop_cube(dataset['cube'], size=cropsize)
        return dataset
    
    cube_route = os.path.join(root, 'center_im.fits')
    cube, header = open_fits(cube_route, header=True)

    if cropsize is not None:
        cube = crop_cube(cube, size=200)

    psf_route = os.path.join(root, 'median_unsat.fits')
    psf       = open_fits(psf_route, header=False)

    ra_route = os.path.join(root, 'rotnth.fits')
    rot_ang  = open_fits(ra_route, header=False)
    rot_ang  = -rot_ang

    filter_name = header['HIERARCH ESO INS COMB IFLT']
    filters = parse_filter_code(filter_name)

    # CENTERING AND SHAPING CUBE
    if cube.shape[-1] % 2 == 0:
        w_cube = []
        for findex, fname in enumerate(filters):
            curr = shift_and_crop_cube(cube[findex], n_jobs=1)
            w_cube.append(curr)
        cube = np.array(w_cube)

    # CENTERING AND SHAPING PSF
    psf_final = []
    for findex, fname in enumerate(filters):
        psf_even = []
        for curr_psf in psf[findex]:
            psf_even.append(modify_shape_and_center(curr_psf, shift_h=1, shift_w=1)) 
        psf_even = np.array(psf_even)
        psf_final.append(psf_even)
    psf_final = np.array(psf_final)

    return {
        'cube': cube,
        'psf': psf_final,
        'q': rot_ang,
        'filters': filters 
    }