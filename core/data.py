import tensorflow as tf
import pandas as pd 
import numpy as np
import os
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from vip_hci.preproc.recentering import frame_shift, frame_center, cube_recenter_2dfit
from photutils.aperture import aperture_photometry, CircularAperture
from vip_hci.preproc.cosmetics import cube_crop_frames
from vip_hci.preproc.recentering import frame_center
from astropy.modeling import models, fitting
from skimage.feature import peak_local_max
from vip_hci.metrics.snr_source	import snr
from vip_hci.var.shapes import get_square
from vip_hci.var import fit_2dgaussian
from multiprocessing import cpu_count
from vip_hci.fm import normalize_psf
from joblib import Parallel, delayed
from vip_hci.psfsub import pca
from astropy.io import fits


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

def dist(yc, xc, y1, x1): #function from vip_hci
    """
    Return the Euclidean distance between two points, or between an array
    of positions and a point.
    """
    return np.sqrt(np.power(yc-y1, 2) + np.power(xc-x1, 2))

def get_intersting_coords(cube, psf_norm, rot_angles, fwhm=4, bkg_sigma = 5, plot=False):
	"""Get coordinates of potential companions
	
	This method infer the background noise and find coordinates that contains most luminous 
	sources. After getting coordinates, it filter them by:
		1. checking that the amplitude is positive > 0
		2. checking whether the x and y centroids of the 2d gaussian fit
		   coincide with the center of the subimage (within 2px error)
		3. checking whether the mean of the fwhm in y and x of the fit are close to 
		   the FWHM_PSF with a margin of 3px
	:param frame: Reduced 2-dim image after applying reduce_pca
	:type frame: np.ndarray
	:param fwhm: full-width at half maximum comming from the normalized PSF, defaults to 4
	:type fwhm: number, optional
	:param bkg_sigma: The number of standard deviations to use for both the lower and upper clipping limit, defaults to 5
	:type bkg_sigma: number, optional
	:param plot: If true, displays original frame vs reconstruction, defaults to False
	:type plot: bool, optional
	:returns: A set of coordinates of potential companions and their associated fluxes
	:rtype: {List of pairs, List of pairs}
	"""

	frame = pca(cube, 
				rot_angles, 
				ncomp=2,
				mask_center_px=None, 
				imlib='opencv', # 'vip-fft'
				interpolation='lanczos4',
				svd_mode='lapack')

	#Calculate sigma-clipped statistics on the provided data.
	_, median, stddev = sigma_clipped_stats(frame, sigma=bkg_sigma, maxiters=None)
	bkg_level = median + (stddev * bkg_sigma)

	# Padding the image with zeros to avoid errors at the edges
	pad_value = 10
	array_padded = np.pad(frame, pad_width=pad_value, mode='constant', constant_values=0)
	
	# plot_to_compare([frame, array_padded], ['Original', 'Padded'])

	# returns the coordinates of local peaks (maxima) in an image.
	coords_temp = peak_local_max(frame, threshold_abs=bkg_level,
								 min_distance=int(np.ceil(fwhm)),
								 num_peaks=20)

	# CHECK BLOBS =============================================================
	y_temp = coords_temp[:, 0]
	x_temp = coords_temp[:, 1]
	coords, fluxes, fwhm_mean = [], [], []

	# Fitting a 2d gaussian to each local maxima position
	for y, x in zip(y_temp, x_temp):
		subsi = 3 * int(np.ceil(fwhm)) # Zone to fit the gaussian
		if subsi % 2 == 0:
		    subsi += 1

		scy = y + pad_value
		scx = x + pad_value

		subim, suby, subx = get_square(array_padded, 
									   subsi, scy, scx,
									   position=True, force=True,
									   verbose=False)
		cy, cx = frame_center(subim)
		gauss = models.Gaussian2D(amplitude=subim.max(), x_mean=cx,
								  y_mean=cy, theta=0,
								  x_stddev=fwhm*gaussian_fwhm_to_sigma,
							      y_stddev=fwhm*gaussian_fwhm_to_sigma)

		sy, sx = np.indices(subim.shape)
		fitter = fitting.LevMarLSQFitter()
		fit = fitter(gauss, sx, sy, subim)
		
		if plot:
			y, x = np.indices(subim.shape)	
			plot_to_compare([subim.T, fit(x, y).T], ['subimage', 'gaussian'], dpi=100, 
				text_box='Companion candidate and its adjusted gaussian model. \
				Here we find an approximated set of coordinates and flux associated to the companion.')

		fwhm_y = fit.y_stddev.value * gaussian_sigma_to_fwhm
		fwhm_x = fit.x_stddev.value * gaussian_sigma_to_fwhm
		mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)])
		# Filtering Process
		condyf = np.allclose(fit.y_mean.value, cy, atol=2)
		condxf = np.allclose(fit.x_mean.value, cx, atol=2)
		condmf = np.allclose(mean_fwhm_fit, fwhm, atol=3)

		aper = CircularAperture((x, y), r=mean_fwhm_fit / 2.) 
		obj_flux_i = aperture_photometry(frame, aper, method='exact')
		obj_flux_i = obj_flux_i['aperture_sum'][0]

		if fit.amplitude.value > 0 and condxf and condyf and condmf:
			coords.append((suby + fit.y_mean.value,
						   subx + fit.x_mean.value))
			fluxes.append(obj_flux_i)
			fwhm_mean.append(mean_fwhm_fit)

	coords = np.array(coords)
	if len(coords) == 0:
		return pd.DataFrame(columns = ['x','y','flux','fwhm_mean'])
	yy = coords[:, 0] - pad_value
	xx = coords[:, 1] - pad_value
	
	table = pd.DataFrame()
	table['x']    = xx
	table['y']    = yy
	table['flux'] = fluxes
	table['fwhm_mean'] = fwhm_mean
	centery, centerx = frame_center(frame)
	drops = []
	for i in range(len(table)):
		sourcex, sourcey = table.x[i], table.y[i]
		sep = dist(centery, centerx, float(sourcey), float(sourcex))
		if not sep > (fwhm / 2) + 1:
			drops.append(table.iloc[i].name)
	table.drop(drops, inplace=True)
	if len(table) > 0:
		table['snr']  = table.apply(lambda col: snr(frame, 
											 	(col['x'], col['y']), 
											 	fwhm, False, verbose=False), axis=1)
	return table, frame

def load_data(root):

	# Complete paths with dafault file names
	cube_route = os.path.join(root, 'center_im.fits')
	psf_route  = os.path.join(root, 'median_unsat.fits')
	rot_route  = os.path.join(root, 'rotnth.fits')

	# Open FITS 
	cube       = fits.getdata(cube_route, ext=0)
	psf_list   = fits.getdata(psf_route, ext=0)
	rot_angles = fits.getdata(rot_route, ext=0)
	rot_angles = -rot_angles

	lambda_ch = np.arange(psf_list.shape[0])
	print('[INFO] channels: ',lambda_ch)

	if type(lambda_ch) == int:
		print(lambda_ch)
		lambda_ch = [lambda_ch]

	new_cube = []
	new_psfs = []
	for lmd_ch in lambda_ch:
		# Store shapes and select wavelenght to work with  
		cube_shp = cube[lmd_ch].shape
		psf_shp  = psf_list[lmd_ch].shape

		# Check multiwave input cube
		if len(cube_shp)<3:
			cube = np.tile(cube[None,...], (psf_list.shape[0], 1, 1, 1))
			cube_shp = cube.shape
			print(cube_shp)

		if len(psf_shp)<3:
			psf_list = psf_list[None, ...]
			psf_shp = psf_list.shape

		# Check even dimensions and correct if found
		if cube_shp[-1] % 2 == 0:
			cube = crop_and_shift(cube[lmd_ch])
		else:
			cube = cube[lmd_ch]

		if psf_shp[-1] % 2 == 0:
			psf = crop_and_shift(psf_list[lmd_ch])
		else:
			psf = psf_list[lmd_ch]

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
		new_cube.append(cube)
		new_psfs.append(psf_norm)

	new_cube = np.array(new_cube)
	new_psfs = np.array(new_psfs)

	return new_cube, new_psfs, rot_angles


def preprocess_fn(folder, root):
	'''Preprocess function'''
	cubes, psfs, rot_angles = load_data(folder)
	for k, (cube, psf) in enumerate(zip(cubes, psfs)):
		table , fr_pca = get_intersting_coords(cube, psf, rot_angles) 
		
		target = os.path.join(root, f'lambda_{k}')
		os.makedirs(target, exist_ok=True)
		table.to_csv(os.path.join(target, 'coords.csv'), index=False)
		for file, file_name in zip([cube, psf, rot_angles, fr_pca], \
		                       ['center_im', 'median_unsat', 'rotnth', 'collapsed_pca']):
			hdu = fits.PrimaryHDU(file)
			hdu.writeto(os.path.join(target, file_name+'.fits'), overwrite=True)

def preprocess_and_save(folder, lambda_ch=0, load_preprocessed=True):

	root = os.path.join(folder, 'processed')
	if not os.path.exists(os.path.join(root, f'lambda_{lambda_ch}')):
		preprocess_fn(folder, root)

	if not load_preprocessed:
		print('[INFO] Preprocessing data...')
		preprocess_fn(folder, root)

	# =================
	cube_route = os.path.join(root, f'lambda_{lambda_ch}', 'center_im.fits')
	psf_route  = os.path.join(root, f'lambda_{lambda_ch}', 'median_unsat.fits')
	rot_route  = os.path.join(root, f'lambda_{lambda_ch}', 'rotnth.fits')

	# Open FITS 
	cube  = fits.getdata(cube_route, ext=0)
	psf   = fits.getdata(psf_route, ext=0)
	rot_angles = fits.getdata(rot_route, ext=0)

	# Open initial very initial guess
	table = pd.read_csv(os.path.join(root, f'lambda_{lambda_ch}', 'coords.csv'))

	return cube, psf, rot_angles, table


def center_frame(frame, x, y, left, bottom, window_size):
	x_sub = x -left
	y_sub = y -bottom
	shift_x = window_size//2 - x_sub 
	shift_y = window_size//2 - y_sub
	centered = frame_shift(frame, shift_y, shift_x, 
	                       imlib='vip-fft', # does a fourier shift operation
	                       interpolation='lanczos4', # Lanczos
	                       border_mode='reflect')
	return centered

def get_companions(cube, x, y, window_size=30, n_jobs=None):
	if n_jobs is None:
		n_jobs = cpu_count()//2
		
	left   = int(x - window_size//2)
	right  = int(x + window_size//2)
	top    = int(y + window_size//2)
	bottom = int(y - window_size//2)

	companion = cube[:, bottom:top, left:right]

	centered = Parallel(n_jobs=n_jobs)(delayed(center_frame)(frame, 
															 x, y, 
															 left, bottom, 
															 window_size) \
								   for frame in companion)

	return np.array(centered, dtype='float32')



def create_tf_dataset(psf, companion, batch_size=1, repeat=1):
	npsf, _, _  = psf.shape
	n_comp, _, _ = companion.shape

	times = int(np.ceil(n_comp/npsf))
	
	psf_ext = [np.tile(psf[i][None,..., None], [times, 1, 1, 1]) for i in range(npsf)]
	psf_ext = np.vstack(psf_ext)[:n_comp]

	inp_x = psf_ext[None,...]
	out_x = companion[None,...,None]

	dataset = tf.data.Dataset.from_tensor_slices((inp_x, out_x))
	dataset = dataset.repeat(repeat)
	dataset = dataset.batch(batch_size)
	return dataset.cache(), psf_ext.shape