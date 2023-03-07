import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
import os
import math

from joblib import Parallel, delayed
from vip_hci.preproc.recentering import frame_shift, frame_center, cube_recenter_2dfit
from vip_hci.preproc.derotation  import frame_rotate, cube_derotate
from vip_hci.var.shapes			 import get_square, prepare_matrix
from vip_hci.preproc.cosmetics 	 import cube_crop_frames
from vip_hci.preproc.parangles   import check_pa_vector
from vip_hci.var 				 import fit_2dgaussian
from vip_hci.fm 				 import normalize_psf
from vip_hci.fits 				 import open_fits
from vip_hci.metrics.snr_source  import snr

from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature 		 import peak_local_max
from scipy.optimize 		 import minimize
from skimage.draw 			 import circle

# Factor with which to multiply Gaussian FWHM to convert it to 1-sigma standard deviation
from astropy.stats 				 import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.modeling 			 import models, fitting
from photutils.centroids 		 import centroid_com
from anotator 					 import AngleAnnotation

COUNT = 0
def shift_and_crop_cube(cube, n_jobs=1, shift_x=-1, shift_y=-1):
	"""Shift and crop each frame within a given cube

	Since VIP only works on odd frames we should rescale if necessary.

	:param cube: Sequence of frames forming the cube
	:type cube: numpy.ndarray
	:param n_jobs: Number of cores to distribute, defaults to 1
	:type n_jobs: number, optional
	:param shift_x: Number of pixels to shift in the x-axis, defaults to -1
	:type shift_x: number, optional
	:param shift_y: Number of pixels to shift in the y-axis, defaults to -1
	:type shift_y: number, optional
	:returns: A recentered and cropped cube containing even-dim frames
	:rtype: {numpy.ndarray}
	"""
	shifted_cube = Parallel(n_jobs=n_jobs)(delayed(frame_shift)(frame, 
	    														shift_x, 
													   			shift_y) \
										   for frame in cube)
	shifted_cube = np.array(shifted_cube)

	y_center, x_center = frame_center(shifted_cube[0])

	ycen   = y_center-0.5
	xcen   = x_center-0.5
	newdim = shifted_cube.shape[-1]-1
	shifted_cube = cube_crop_frames(shifted_cube,
	                                newdim,
	                                xy=[int(ycen), int(xcen)], 
	                                force=True) 
	return shifted_cube
	

def fit_and_crop(cube, use_pos=0, n_jobs=1):
	"""Fit a Gaussian and crop an frame-based cube.
	
	This method is an alternative to 'shift_and_crop_cube'. 
	Dimensions are reduced in one pixel to satisfies the VIP requirements.
	A gaussian model is trained on one frame, then the adjusted parameters 
	are used to fit another gaussian along the cube frames.

	:param cube: A cube containing frames
	:type cube: numpy.ndarray
	:param use_pos: Reference frame, defaults to 0
	:type use_pos: number, optional
	:param n_jobs: Number of cores to distribute, defaults to 1
	:type n_jobs: number, optional
	:returns: A recentered and cropped cube containing even-dim frames
	:rtype: {numpy.ndarray}
	"""
	# Fit a 2-dim Gaussian to the cropped PSF image 
	model_2d = fit_2dgaussian(cube[use_pos, :-1, :-1], 
							  crop=True, 
							  cropsize=30, 
							  debug=True, 
							  full_output=True)

	fwhm_sphere = np.mean([model_2d.fwhm_y, model_2d.fwhm_x]) 
	y_c, x_c = frame_center(cube[use_pos, :-1, :-1]) 
	cube_center, y0_sh, x0_sh = cube_recenter_2dfit(cube, 
	                                               (int(y_c), int(x_c)), 
	                                               fwhm_sphere, 
	                                               model='gauss',
	                                               nproc=n_jobs, 
	                                               subi_size=7,
	                                               negative=False, #?
	                                               full_output=True, 
	                                               debug=False,
	                                               plot=False,
	                                               verbose=False) 

	return cube_center, fwhm_sphere, model_2d

def plot_to_compare(images, titles, axes=None, show=True, img_file=None, **savefig_params):
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
	if axes is None:
		fig, axes = plt.subplots(1, len(images), dpi=300,
			gridspec_kw={'hspace': 0., 'wspace': .4})
	for i, (im, ti) in enumerate(zip(images, titles)):
		im_obj = axes[i].imshow(im)
		# axes[i].set_ylim(50, 150)
		# axes[i].set_xlim(50, 150)
		divider = make_axes_locatable(axes[i])
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(im_obj, cax=cax)
		axes[i].set_title(ti)
	if show:
		plt.show()
	if img_file:
		fig.savefig(img_file, **savefig_params)

	return axes 

def plot_cube(cube, save=False):
	""" Plot each frame from a cube
	
	:param cube: A cube containing frames
	:type cube: numpy.ndarray
	:param save: Write each frame figure, defaults to False
	:type save: bool, optional
	"""
	for i in range(cube[0].shape[0]):
		fig, axes = plt.subplots(1, 2, dpi=300,
		gridspec_kw={'hspace': 0., 'wspace': .4})		
		for k in range(2):
			y, x  = frame_center(cube[k][i])
			frame = get_square(cube[k][i], size=40, y=y, x=x, position=False)
			im_obj = axes[k].imshow(np.log(frame))
			divider = make_axes_locatable(axes[k])
			cax = divider.append_axes("right", size="5%", pad=0.05)
			plt.colorbar(im_obj, cax=cax)

		axes[0].set_title(r'$\lambda = H2$')
		axes[1].set_title(r'$\lambda = H1$')
		fig.text(.38, .85, f'{i}-th frame from the cube', va='center', rotation='horizontal')
		if save:
			plt.savefig(f'./figures/cube_gif/{i}.png', format='png',  bbox_inches = "tight")
		else:
			plt.show()
			

def fit_gaussian_2d(image, fwhmx=4, fwhmy=4, plot=False):
	""" Fit a 2 dimensional gaussian
	
	This method first creates a gaussian model from the parameters of images. 
	Then it adjusts the gaussian parameters using Levenberg-Marquardt algorithm.
	:param image: 2D image
	:type image: numpy.ndarray
	:returns: full-width at half maximum in x/y axes and the mean x/y after fitting
	:rtype: {number, number, number, number}
	"""

	init_amplitude = np.ptp(image) # Max diff
	# Calculate the centroid of an n-dimensional array as its “center of mass” determined from moments.
	xcom, ycom = centroid_com(image) 

	# Creates a gaussian model
	gauss = models.Gaussian2D(amplitude=init_amplitude, # Amplitude (peak value) of the Gaussian.
							  theta=0., # The rotation angle as an angular quantity or a value in radians. 
	                          x_mean=xcom, y_mean=ycom, # Mean of the Gaussian
	                          x_stddev=fwhmx * gaussian_fwhm_to_sigma,
	                          y_stddev=fwhmy * gaussian_fwhm_to_sigma)

	# Levenberg-Marquardt algorithm and least squares statistic.
	fitter = fitting.LevMarLSQFitter()
	y, x = np.indices(image.shape)
	fit = fitter(gauss, x, y, image)

	mean_y = fit.y_mean.value
	mean_x = fit.x_mean.value
	fwhm_y = fit.y_stddev.value*gaussian_sigma_to_fwhm # going back to original fwhm
	fwhm_x = fit.x_stddev.value*gaussian_sigma_to_fwhm # going back to original fwhm
	amplitude = fit.amplitude.value
	theta = np.rad2deg(fit.theta.value) # radian to degree

	fitter.fit_info['param_cov'] # uncertanties == standard deviation
	perr = np.sqrt(np.diag(fitter.fit_info['param_cov']))
	amplitude_e, mean_x_e, mean_y_e, fwhm_x_e, fwhm_y_e, theta_e = perr
	fwhm_x_e /= gaussian_fwhm_to_sigma
	fwhm_y_e /= gaussian_fwhm_to_sigma

	if plot:
		plot_to_compare([image, fit(x, y)], ['Original', 'Model'])
	return fwhm_y, fwhm_x, mean_y, mean_x

def recenter_cube(cube, ref_frame, fwhm_sphere=4, subi_size=7, n_jobs=1):
	"""Recenter a cube of frames based on a frame reference
	
	Using the estimated FWHM we fit gaussian models to center a sequence of frames 
	:param cube: A cube containing frames
	:type cube: numpy.ndarray
	:param ref_frame: A reference frame (e.g., the first PSF)
	:type ref_frame: numpy.ndarray
	:param fwhm_sphere: Full-Width at Half Maximum value to initialice the gaussian model, defaults to 4
	:type fwhm_sphere: number, optional
	:param subi_size: Size of the square subimage sides in pixels. must be even, defaults to 7
	:type subi_size: number, optional
	:returns: A recentered cube of frames
	:rtype: {numpy.ndarray}
	"""
	n_frames, sizey, sizex = cube.shape
	fwhm 		 = np.ones(n_frames) * fwhm_sphere
	pos_y, pos_x = frame_center(ref_frame)
	psf_rec 	 = np.empty_like(cube) # template for the reconstruction

	# ================================================================================================================
	# Function to distribute (i.e., what it is inside the for loop)
	def cut_fit_shift(frame, size, fwhm, y, x, plot=False):
		# Cut the frame en put the center in the (x,y) from the reference frame
		sub, y1, x1 = get_square(frame, size=size, y=y, x=x, position=True)

		if plot:
			# [Only for visualization] Negative gaussian fit
			sub_to_plot = sub
			sub_image = -sub + np.abs(np.min(-sub))
			plot_to_compare([sub_to_plot, sub_image], ['Original', 'Negative'])

		_, _, y_i, x_i = fit_gaussian_2d(sub, fwhmx=fwhm, fwhmy=fwhm)
		y_i += y1
		x_i += x1
		# 
		shift_x = x - x_i
		shift_y = y - y_i 
		# Going back to the original frame
		centered = frame_shift(frame, shift_y, shift_x, 
							   imlib='vip-fft', # does a fourier shift operation
							   interpolation='lanczos4', # Lanczos
							   border_mode='reflect') # input extended by reflecting about the edge of the last pixel.
		return centered
	# ================================================================================================================
	# Iterates over the PSF in a distributed manner
	shifted_cube = Parallel(n_jobs=n_jobs)(delayed(cut_fit_shift)(cube[fnum],
	    														  size=subi_size,
	    														  fwhm=fwhm[fnum], 
													   			  y=pos_y, x=pos_x,
													   			  plot=False) \
										   for fnum in range(n_frames))
	shifted_cube = np.array(shifted_cube) # put all together as a numpy array

	return shifted_cube

def reduce_pca(cube, rot_angles, ncomp=1, fwhm=4, plot=False, n_jobs=1):
	""" Reduce cube using Angular Differential Imaging (ADI) techinique. 
	
	This function reduce the frame-axis dimension of the cube to 1
	We subtract the principal components to each original frame (residuals).
	Using the rotation angles we center all the frames and reduce them to the median. 
	If more than one channels were provided, we calculate the mean of the medians per wavelength.
	:param cube: A cube containing frames
	:type cube: numpy.ndarray
	:param rot_angles: Rotation angles used to center residual frames
	:type rot_angles: numpy.ndarray
	:param ncomp: Number of component to reduce in the frames axis, defaults to 1
	:type ncomp: number, optional
	:param fwhm_sphere: Full-Width at Half Maximum value to initialice the gaussian model, defaults to 4
	:type fwhm_sphere: number, optional
	:param plot: If true, displays original frame vs reconstruction, defaults to False
	:type plot: bool, optional
	:returns: Median of centered residuals
	:rtype: {np.ndarray}
	"""
	nz, ny, nx = cube.shape
	rot_angles = check_pa_vector(rot_angles)

	# Build the matrix for the SVD/PCA and other matrix decompositions. (flatten the cube)
	matrix = prepare_matrix(cube, mode='fullfr', verbose=False)

	# DO SVD on the matrix values
	U, S, V = np.linalg.svd(matrix.T, full_matrices=False)
	# `matrix.T` has dimension LxD where L is `time steps` and D is `pixels space`
	# We want to see the max. var or info in the pixels space along the time steps axis
	# Then when applying SVD,
	# The columns of U represent the principal components along the time steps.
	# The columns of V represent the principal components along the pixel space.
	U = U[:, :ncomp].T

	transformed   = np.dot(U, matrix.T)
	reconstructed = np.dot(transformed.T, U)
	residuals     = matrix - reconstructed

	residuals 	  = residuals.reshape(residuals.shape[0], ny, nx)
	reconstructed = reconstructed.reshape(reconstructed.shape[0], ny, nx)
	matrix 		  = matrix.reshape(matrix.shape[0], ny, nx)
	p_components  = U.reshape(U.shape[0], ny, nx)

	# NOT SURE WHY rot_angles IS NEGATIVE
	array_der = cube_derotate(residuals, -rot_angles, nproc=n_jobs, 
							  imlib='vip-fft', interpolation='lanczos4')
	res_frame = np.nanmedian(array_der, axis=0)
	if plot or True:
		plot_to_compare([matrix[0], reconstructed[0], residuals[0], res_frame], 
						['Original', 'Reconstructed', 'Residuals', 'median'])

	return res_frame

def get_intersting_coords(frame, psf_norm, fwhm=4, bkg_sigma = 5, plot=False):
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
	#Calculate sigma-clipped statistics on the provided data.
	_, median, stddev = sigma_clipped_stats(frame, sigma=bkg_sigma, maxiters=None)
	bkg_level = median + (stddev * bkg_sigma)
	threshold = bkg_level

	from scipy.ndimage.filters import correlate
	frame_det = correlate(frame, psf_norm)

	# returns the coordinates of local peaks (maxima) in an image.
	coords_temp = peak_local_max(frame_det, threshold_abs=bkg_level,
								 min_distance=int(np.ceil(fwhm)),
								 num_peaks=20)

	# Padding the image with zeros to avoid errors at the edges
	pad_value = 10
	array_padded = np.pad(frame, pad_width=pad_value, mode='constant', constant_values=0)
	if plot:
		plot_to_compare([frame, array_padded], ['Original', 'Padded'])

	# ===================================================================================
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
			plot_to_compare([subim, fit(x, y)], ['subimage', 'gaussian'])

		fwhm_y = fit.y_stddev.value * gaussian_sigma_to_fwhm
		fwhm_x = fit.x_stddev.value * gaussian_sigma_to_fwhm
		mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)])
		# Filtering Process
		condyf = np.allclose(fit.y_mean.value, cy, atol=2)
		condxf = np.allclose(fit.x_mean.value, cx, atol=2)
		condmf = np.allclose(mean_fwhm_fit, fwhm, atol=3)
		if fit.amplitude.value > 0 and condxf and condyf and condmf:
			coords.append((suby + fit.y_mean.value,
						   subx + fit.x_mean.value))
			fluxes.append(fit.amplitude.value)
			fwhm_mean.append(mean_fwhm_fit)

	coords = np.array(coords)
	yy = coords[:, 0] - pad_value
	xx = coords[:, 1] - pad_value
	
	table = pd.DataFrame()
	table['x']    = xx
	table['y']    = yy
	table['flux'] = fluxes
	table['fwhm_mean'] = fwhm_mean
	table['snr']  = table.apply(lambda col: snr(frame, 
											 	(col['x'], col['y']), 
											 	fwhm, False, verbose=False), axis=1)

	return table
def run_pipeline(cube_path, psf_path, rot_ang_path, wavelength=0, psf_pos=0, 
				pixel_scale=0.01225, n_jobs=1, plot=False):
	"""Main function to run Negative Fake Companion (NEGFC) preprocessing.
	
	:param cube_path: Path to the cube image
	:type cube_path: string
	:param psf_path: Path to the PSF image
	:type psf_path: string
	:param rot_ang_path: Path to the rotation angles
	:type rot_ang_path: string
	:param wavelength: Wavelength to use (H2=0  H3=1 / K1=0  K2=1), defaults to 0
	:type wavelength: number, optional
	:param psf_pos: Either the initial (0) or final (1) PSF, defaults to 0
	:type psf_pos: number, optional
	:param pixel_scale: pixel scale arcsec/pixel, defaults to 0.01225 from IRDIS/SPHERE
	:type pixel_scale: number, optional
	"""
		
	# First we load images from paths
	cube       = open_fits(cube_path,    header=False) 
	psf        = open_fits(psf_path,     header=False) 
	rot_angles = open_fits(rot_ang_path, header=False) # where they come from?
	# plot_cube(cube, save=True)
	# plot_to_compare([psf[1][0], psf[1][1]], ['PSF init', 'PSF end'])

	# Check cube dimensions
	if cube.shape[-1] % 2 == 0:
		print('[WARNING] Cube contains odd frames. Shifting and rescaling...')
		cube = shift_and_crop_cube(cube[wavelength], n_jobs=n_jobs)

	single_psf = psf[wavelength, psf_pos, :-1, :-1]
	ceny, cenx = frame_center(single_psf)
	imside = single_psf.shape[0]
	cropsize = 30

	psf_subimage, suby, subx = get_square(single_psf, 
										  min(cropsize, imside),
	                                      ceny, cenx, 
	                                      position=True, 
	                                      verbose=False)
	if plot:
		plot_to_compare([single_psf, psf_subimage], ['Original', 'Subimage'])

	fwhm_y, fwhm_x, mean_y, mean_x = fit_gaussian_2d(psf_subimage, plot=plot)
	mean_y +=  suby # put the subimage in the original center
	mean_x +=  subx # put the subimage in the original center
	
	fwhm_sphere  = np.mean([fwhm_y, fwhm_x]) # Shared across the frames 
	psf_rec = recenter_cube(psf[wavelength], 
							single_psf, 
							fwhm_sphere=fwhm_sphere, 
							n_jobs=n_jobs)
	
	# Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM aperture equal to one. 
	# It also allows to crop the array and center the PSF at the center of the array(s).
	psf_norm, fwhm_flux, fwhm = normalize_psf(psf_rec[psf_pos], 
	                                          fwhm=fwhm_sphere,
	                                          full_output=True, 
	                                          verbose=False) 
	if plot:
		plot_to_compare([psf_rec[psf_pos], psf_norm], ['PSF reconstructed', 'PSF normalized'])

	# ======== MOON DETECTION =========
	frame = reduce_pca(cube[wavelength], rot_angles, ncomp=1, fwhm=4, plot=plot, n_jobs=n_jobs)
	# Blob can be defined as a region of an image in which some properties are constant or 
	# vary within a prescribed range of values.
	table = get_intersting_coords(frame, psf_norm, fwhm=fwhm, bkg_sigma=5, plot=plot)
	# remove coords having low signal to noise ratio
	snr_thresh = 2
	table = table[table['snr'] > snr_thresh]

	# How many FWHM we want to consider to fit the model
	nfwhm = 3
	fwhma = int(nfwhm)*float(fwhm_sphere)
	# Cube to store the final model
	cube_emp = np.zeros(cube[0].shape)
	x_cube_center, y_cube_center = frame_center(cube[wavelength, 0, ...])

	# Detection from coords NegFC
	nframes = cube[wavelength].shape[0]
	for _, row in table.iterrows():

		# fig, ax = plt.subplots(figsize=(5,5), dpi=200)
		# ax.imshow(frame)
		# for _, row in table.iterrows():
		# 	circle = plt.Circle((row['x'], row['y']), row['fwhm_mean'], fill=False, edgecolor='r')	
		# 	ax.add_patch(circle)
		# 	ax.set_xlim(50, 150)
		# 	ax.set_ylim(50, 150)
		# plt.show()
		# return

		for index in range(nframes): 
			x=float(row['x']) - x_cube_center
			y=float(row['y']) - y_cube_center

			current_frame = cube[wavelength, index]

			radius = np.sqrt(x**2+y**2) # radius
			angle  = np.arctan2(y, x)   # radians
			angle  = angle*180/np.pi    # degrees
			# Since atan2 return angles in between [0, 180] and [-180, 0],
			# we convert the angle to refers a system of 360 degrees
			theta0 = np.mod(angle, 360) 

			params = (radius, theta0, table['flux'].values)

			# sub, _, _ = get_square(current_frame, 
			# 					   size=50, 
			# 					   y=y_cube_center, 
			# 					   x=x_cube_center, 
			# 					   position=True)

			# plot_angles(sub, x,	y, index)
			
			# solu = minimize(chisquare_mod, params, 
			# 				args=(row['x'], row['y'], 
			# 					  current_frame, 
			# 					  rot_angles[index], 
			# 					  pixel_scale, 
			# 					  psf_norm, 
			# 					  fwhma, 
			# 					  'stddev'),
            #     			method = 'Nelder-Mead')

def plot_angles(sub, x, y, index=0):
	fig, ax = plt.subplots(nrows=1, ncols=1)
	ax.imshow(sub, extent=[-sub.shape[1]/2., 
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


def chisquare_mod(modelParameters, sourcex, sourcey, frame, ang, 
				   pixel, psf_norma, fwhm, fmerit):
	"""Creates the objetive function to be minimized

	This function calculate residuals (errors) based on first guesses 
	of the physical parameters of the companion candidate.
	:param modelParameters: Parameters to be adjusted
	:type modelParameters: list of scalars
	:param sourcex: coordinate in axis x
	:type sourcex: number, float
	:param sourcey: coordinate in axis y
	:type sourcey: number, float
	:param frame: Current 2-dimensional frame from the cube
	:type frame: numpy.ndarray
	:param ang: Rotation angle
	:type ang: number, float
	:param pixel: Pixel scale
	:type pixel: number, float
	:param psf_norma: Normalized PSF
	:type psf_norma: numpy.ndarray
	:param fwhm: Full width at Half Maximum to consider during the optimization
	:type fwhm: number, float
	:param fmerit: How to calculate residuals. Either taking the 'stddev' or the 'sum'
	:type fmerit: string
	:returns: loss to minimize
	:rtype: {number, float}
	"""
	try:
		r, theta, flux = modelParameters
	except TypeError:
		print('paraVector must be a tuple, {} was given'.format(type(modelParameters)))

	frame_negfc = inject_fcs_cube_mod(frame, 
									  psf_norma, 
									  ang, 
									  -flux, 
									  r, 
									  theta, 
									  n_branches=1)
	

	centy_fr, centx_fr = frame_center(frame_negfc)

	posy = r * np.sin(np.deg2rad(theta)-np.deg2rad(ang)) + centy_fr
	posx = r * np.cos(np.deg2rad(theta)-np.deg2rad(ang)) + centx_fr

	indices = circle(posy, posx, radius=fwhm)
	yy, xx = indices
	values = frame_negfc[yy, xx].ravel()    

	# Function of merit
	if fmerit == 'sum':
		values = np.abs(values)
		chi2 = np.sum(values[values > 0])
		N = len(values[values > 0])
		loss =  chi2 / (N-3) 
	if fmerit == 'stddev':
	    loss = np.std(values[values != 0]) # loss
	
	fig, axes = plt.subplots(1, 3, figsize=(5,5), sharex=True, sharey=True, dpi=300)
	axes = axes.flatten()
	axes[0].imshow(frame)
	axes[0].set_title('Frame')
	axes[0].set_ylim(50, 150)
	axes[0].set_xlim(50, 150)
	axes[1].imshow(frame_negfc)
	axes[1].set_title('Frame \n+ Fake Companion')
	axes[1].set_ylim(50, 150)
	axes[1].set_xlim(50, 150)
	axes[2].imshow(frame_negfc-frame)
	axes[2].set_title('Residuals {:.2f}'.format(loss))
	axes[2].set_ylim(50, 150)
	axes[2].set_xlim(50, 150)
	root = './figures/negfc_opt/'
	files = os.listdir(root)
	if len(files) == 0:
		fig.savefig(root+'0.png')
	else:
		numbers = [int(file.split('.png')[0]) for file in files]
		numbers = np.sort(numbers)
		fig.savefig(root+'{}.png'.format(numbers[-1]+1))

	return loss


def inject_fcs_cube_mod(frame, template, angle, flux, radius, 
                    	theta, n_branches=1, imlib='opencv'):
	"""Inject a template image into a frame
	
	Template usually is a PSF and the frame cames from the cube.
	:param frame: Frame where we are going to inject the template
	:type frame: numpy.ndarray
	:param template: Template or patch to be injected into the frame
	:type template: numpy.ndarray
	:param angle: Rotation angle of the frame 
	:type angle: number, float
	:param flux: Flux guess of the companion
	:type flux: number, float
	:param radius: Distance between the planet and the companion
	:type radius: number, float
	:param theta: Angle in degrees between the star and the companion 
	:type theta: number, float
	:param n_branches: [description], defaults to 1
	:type n_branches: number, optional
	:param imlib: image library to work with, defaults to 'opencv'
	:type imlib: str, optional
	:returns: [description]
	:rtype: {[type]}
	"""
	ceny, cenx = frame_center(frame)
	size_psf = template.shape[0]

	# inyect the PSF template to the center of the image
	frame_copy = np.zeros_like(frame, dtype=np.float64)
	w = int(np.floor(size_psf/2.)) # width
	frame_copy[int(ceny-w):int(ceny+w+1), int(cenx-w):int(cenx+w+1)] = template

	# Here we insert many companions around the center of the image. 
	# Since we already know an approximated set of coordinates of the companions,
	# we do not need to inject more than one fake companion around the center of the image. 
	# That means in we'll only use the theta angle for injecting (i.e., branch = 0)	
	tmp = np.zeros_like(frame)
	for branch in range(n_branches):
		ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta)
		y = radius * np.sin(ang - np.deg2rad(angle))
		x = radius * np.cos(ang - np.deg2rad(angle))
		# we shape the normed PSF to the companion flux
		img_shifted = frame_shift(frame_copy, y, x, imlib=imlib)
		tmp += img_shifted*flux
	frame_out = frame + tmp
	return frame_out   

if __name__ == '__main__':

	# # Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM aperture equal to one. 
	# # It also allows to crop the array and center the PSF at the center of the array(s).
	# # QUESTIONS:
	# # 1) Can I normalice both PSFs using the same fitted model? is it worth?
	# # 2) Can I get FWHM directly from the pixel distribution WITHOUT FITTING A GAUSSIAN?

	cube_path    = './data/HCI/center_im.fits'
	psf_path     = './data/HCI/median_unsat.fits' # Why this name?
	rot_ang_path = './data/HCI/rotnth.fits'


	run_pipeline(cube_path, psf_path, rot_ang_path, 
		n_jobs=5, plot=False)