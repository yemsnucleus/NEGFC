import pandas as pd
import numpy as np
import os

from multiprocessing import cpu_count
from astropy.io import fits


from vip_hci.preproc.recentering import frame_shift, frame_center, cube_recenter_2dfit
from vip_hci.preproc.cosmetics import cube_crop_frames
from vip_hci.var import fit_2dgaussian
from vip_hci.fm import normalize_psf
from vip_hci.psfsub import pca
from vip.detection import get_intersting_coords

def get_radius_theta(row, cube_shp):
    height, width = cube_shp
    
    x = float(row['x']) - height//2
    y = float(row['y']) - width//2
    
    radius = np.sqrt(x**2+y**2) # radius
    angle  = np.arctan2(y, x)   # radians
    angle  = angle/np.pi*180    # degrees
    # Since atan2 return angles in between [0, 180] and [-180, 0],
    # we convert the angle to refers a system of 360 degrees
    theta0 = np.mod(angle, 360) 
    
    row['theta'] = theta0
    row['radius'] = radius
    return row

def crop_and_shift(inputs):
	shape = inputs.shape
	shifted=np.zeros_like(inputs)
	for i in range (shape[0]):
		shifted[i,:,:] = frame_shift(inputs[i,:,:], -1, -1)  
	yc, xc = frame_center(inputs)
	xcen, ycen = xc-0.5, yc-0.5
	newdim= shape[-1]-1
	cropped = cube_crop_frames(shifted, newdim, xy=[int(ycen), int(xcen)], force=True)  
	return cropped

def load_data(root, lambda_ch=0):
	# Complete paths with dafault file names
	cube_route = os.path.join(root, 'center_im.fits')
	psf_route  = os.path.join(root, 'median_unsat.fits')
	rot_route  = os.path.join(root, 'rotnth.fits')

	# Open FITS 
	cube       = fits.getdata(cube_route, ext=0)
	psf_list   = fits.getdata(psf_route, ext=0)
	rot_angles = fits.getdata(rot_route, ext=0)
	rot_angles = -rot_angles

	# Store shapes and select wavelenght to work with  
	cube_shp = cube[lambda_ch].shape
	psf_shp  = psf_list[lambda_ch].shape

	# Check multiwave input cube
	if len(cube_shp)<3:
		cube = cube[None, ...]
		cube_shp = cube.shape
	if len(psf_shp)<3:
		psf_list = psf_list[None, ...]
		psf_shp = psf_list.shape

	# Check even dimensions and correct if found
	if cube_shp[-1] % 2 == 0:
		cube = crop_and_shift(cube[lambda_ch])
	else:
		cube = cube[lambda_ch]

	if psf_shp[-1] % 2 == 0:
		psf = crop_and_shift(psf_list[lambda_ch])
	else:
		psf = psf_list[lambda_ch]

	# PSF Normalization
	fit = fit_2dgaussian(psf[0], 
						 crop=True, 
						 cropsize=30, 
						 debug=False, 
						 full_output=True)   

	fwhm_sphere = np.mean([fit.fwhm_y, fit.fwhm_x])
	y_cent, x_cent = frame_center(psf)
	y_c=int(y_cent)
	x_c=int(x_cent)
	psf_center, y0_sh, x0_sh = cube_recenter_2dfit(psf, 
												   (y_c, x_c), 
												   fwhm_sphere,
												   model='gauss',
												   nproc=cpu_count()//2, 
												   subi_size=7, 
												   negative=False,
												   full_output=True, 
												   plot=False,
												   debug=False)  

	psf_norm = normalize_psf(psf_center, 
							fwhm=fwhm_sphere, 
							size=None, 
							threshold=None, 
							mask_core=None,
 						    full_output=False, 
 						    verbose=False) 
	return cube, psf_norm, rot_angles

def find_closest_row_index(x_value, y_value, df_true):
    """Finds the index of the row in another dataset that has the closest value x, y with the other one."""
    distances = np.sqrt(((df_true['true_x'] - x_value)**2) + ((df_true['true_y'] - y_value)**2))
    closest_row_index = distances.argmin()
    return closest_row_index

def preprocess_folder(root, target_folder):
    if not os.path.isdir(target_folder):
        print('[INFO] Preprocessing data')
        cube, psf, rot_angles = load_data(root)
        fr_pca = pca(cube, 
                     rot_angles, 
                     ncomp=5,
                     mask_center_px=None, 
                     imlib='opencv', # 'vip-fft'
                     interpolation='lanczos4',
                     svd_mode='lapack')

        table = get_intersting_coords(fr_pca, psf, fwhm=4, bkg_sigma=3)

        os.makedirs(target_folder, exist_ok=True)

        for file, file_name in zip([cube, psf, rot_angles, fr_pca], ['center_im', 'median_unsat', 'rotnth', 'collapsed_pca']):
            hdu = fits.PrimaryHDU(file)
            hdu.writeto(os.path.join(target_folder, file_name+'.fits'), overwrite=True)
            
            
        true_table = pd.read_csv(os.path.join(root, 'true_values.csv'))
        closest_indices = []
        for index, row in table.iterrows():
            closest = find_closest_row_index(row['x'], row['y'], true_table) 
            ctt = true_table.iloc[closest]
            closest_indices.append(ctt)

        final = pd.DataFrame(closest_indices).reset_index()
        final = pd.concat([table, final], axis=1, ignore_index=False, sort=False)
    
    
        final.to_csv(os.path.join(target_folder, 'init_params.csv'), index=False)
    else:
        print('[INFO] Restoring saved values')
        # Complete paths with dafault file names
        cube_route = os.path.join(target_folder, 'center_im.fits')
        psf_route  = os.path.join(target_folder, 'median_unsat.fits')
        rot_route  = os.path.join(target_folder, 'rotnth.fits')

        # Open FITS 
        cube  = fits.getdata(cube_route, ext=0)
        psf   = fits.getdata(psf_route, ext=0)
        rot_angles = fits.getdata(rot_route, ext=0)

        # Open initial very initial guess
        table = pd.read_csv(os.path.join(target_folder, 'init_params.csv'))

    table = table.apply(lambda x: get_radius_theta(x, cube.shape[1:]), 1)

    return cube, psf, rot_angles, table

