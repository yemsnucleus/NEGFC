import gpu.fake_comp as tfnegfc
import tensorflow as tf
import os

from .fake_comp import create_patch
from astropy.io import fits


def format_input(xy, flux, cube, psf, rot):
    inputs = {
        'psf': psf,
        'rot_angles':rot,
    }
    
    outputs = {
        'cube':cube,
        'rot_angles':rot,
    }
    
    return inputs, outputs

def get_dataset(xy_pos, flux, cube, psf, rot_ang, lambda_ch=0, psf_pos=0):    
    psf = create_patch(cube[lambda_ch, psf_pos], psf[lambda_ch])

    cube_inp = cube[lambda_ch]

    dataset = tf.data.Dataset.from_tensor_slices((xy_pos[None,...],
                                                  flux[None,...],
                                                  cube_inp[None,...], 
                                                  psf[None,...], 
                                                  rot_ang[None,...]))
    dataset = dataset.map(format_input)
    return dataset.batch(1)

def load_data(root, lambda_ch = 0, psf_pos=0, ncomp=1):
    cube_route = os.path.join(root, 'center_im.fits')
    cube  = fits.getdata(cube_route, ext=0)
#     cube = cube[None,...]
    
    psf_route  = os.path.join(root, 'median_unsat.fits')
    psf  = fits.getdata(psf_route, ext=0)
    
    ra_route   = os.path.join(root, 'rotnth.fits')
    rot_ang    = fits.getdata(ra_route, ext=0)
    rot_ang    = -rot_ang
    
    # NORMALIZE PSF
    results = tfnegfc.adjust_gaussian(psf[lambda_ch, psf_pos])
    fwhm_sphere  = tf.reduce_mean(results['fwhm'])
    centered_psf = tfnegfc.center_cube(psf[lambda_ch], fwhm_sphere)
    normalized_psf = tfnegfc.normalize_psf(centered_psf, fwhm=fwhm_sphere)
    
    # GET CANDIDATES
    adi_image, res_cube = tfnegfc.apply_adi(cube[lambda_ch], 
                                        rot_ang, 
                                        out_size=cube[lambda_ch].shape, 
                                        ncomp=ncomp, 
                                        derotate='tf', 
                                        return_cube=True)
    
    table = tfnegfc.get_coords(adi_image.numpy(), 
                               fwhm=fwhm_sphere, 
                               bkg_sigma=5, 
                               cut_size=10)
    
    xy_cords  = table[['x', 'y']].values
    init_flux = table['flux'].values 
    
    
    dataset = get_dataset(xy_cords,
                          init_flux,
                          cube, 
                          normalized_psf, 
                          rot_ang, 
                          lambda_ch=lambda_ch, 
                          psf_pos=psf_pos)
    
    return dataset, cube[lambda_ch].shape, xy_cords, init_flux
