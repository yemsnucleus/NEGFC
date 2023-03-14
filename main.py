# =============================
# By YEMS EXO-MOONS GROUP 2023
# =============================
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import math
import os

from vip_hci.preproc.recentering 	import frame_shift, frame_center, cube_recenter_2dfit
from vip_hci.preproc.derotation  	import frame_rotate
from vip_hci.preproc.cosmetics 	 	import cube_crop_frames
from vip_hci.var 				 	import fit_2dgaussian
from vip_hci.fm 				 	import normalize_psf
from vip_hci.var.shapes				import get_square
from vip_hci.fits 					import open_fits

from plottools					import plot_to_compare, plot_cube, plot_detection, plot_cube_multiband 
from detection 					import get_intersting_coords, optimize_params
from pca 						import reduce_pca

from astropy.stats				import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm

from joblib						import Parallel, delayed
from astropy.modeling			import models, fitting
from photutils.centroids		import centroid_com


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

def fit_gaussian_2d(image, fwhmx=4, fwhmy=4, plot=False, dpi=100, text_box=''):
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
		plot_to_compare([image, fit(x, y)], ['Original', 'Model'], dpi=dpi, text_box=text_box)
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
		sub, y1, x1 = get_square(frame, size=size, y=y, x=x, position=True, force=True)

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

def run_pipeline(opt):
	"""Main function to run Negative Fake Companion (NEGFC) preprocessing."""
		
	# First we load images from paths
	cube        = open_fits(opt.cube,    header=False) 
	psf         = open_fits(opt.psf,     header=False) 
	rot_angles  = open_fits(opt.ra, 	 header=False)
	pixel_scale = opt.px_corr

	rot_angles  = -rot_angles + opt.ang_corr
	if opt.plot:
		plot_cube_multiband(cube, dpi=100, save=False, text_box='Frame cubes separated by wavelengths. \
															     We have 90 frames in total.')
		# plot_to_compare([psf[1][0], psf[1][1]], ['PSF init', 'PSF end'])

	# Check cube dimensions
	if cube.shape[-1] % 2 == 0:
		print('[WARNING] Cube contains odd frames. Shifting and rescaling...')
		cube = shift_and_crop_cube(cube[opt.w], n_jobs=opt.njobs)

	single_psf = psf[opt.w, opt.p, :-1, :-1]
	ceny, cenx = frame_center(single_psf)
	imside = single_psf.shape[0]
	cropsize = 30

	psf_subimage, suby, subx = get_square(single_psf, 
										  min(cropsize, imside),
	                                      ceny, cenx, 
	                                      position=True, 
	                                      verbose=False)
	if opt.plot:
		plot_to_compare([single_psf, psf_subimage], ['Original', 'Subimage'], 
						text_box='Original and cropped image. \
								We cut the original PSF image to avoid \
								processing void pixels around the star')

	fwhm_y, fwhm_x, mean_y, mean_x = fit_gaussian_2d(psf_subimage, plot=opt.plot, dpi=100, 
		text_box='Gaussian model adjusted on the original PSF (LEFT) pixels. \
				  We fit a parametric models to find the actual center of the star')

	mean_y +=  suby # put the subimage in the original center
	mean_x +=  subx # put the subimage in the original center
	
	fwhm_sphere  = np.mean([fwhm_y, fwhm_x]) # Shared across the frames 
	psf_rec = recenter_cube(psf[opt.w], 
							single_psf, 
							fwhm_sphere=fwhm_sphere, 
							n_jobs=opt.njobs)
	
	# Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM aperture equal to one. 
	# It also allows to crop the array and center the PSF at the center of the array(s).
	psf_norm, fwhm_flux, fwhm = normalize_psf(psf_rec[opt.p], 
	                                          fwhm=fwhm_sphere,
	                                          full_output=True, 
	                                          verbose=False) 
	if opt.plot:
		plot_to_compare([psf_rec[opt.p], psf_norm], ['PSF reconstructed', 'PSF normalized'], dpi=100,
					text_box='Normalized PSF. We use the normalized PSF as a mold to learn the distribution of the companion.')

	# ======== MOON DETECTION =========
	frame, res_cube = reduce_pca(cube[opt.w], rot_angles, ncomp=1, fwhm=4, plot=opt.plot, 
								return_cube=True, dpi=100, n_jobs=opt.njobs,
								text_box='Collapsed (median) PCA residuals after using the first principal component \
								along the time axis to reconstruct the original image. The first 3 images correspond to a single frame-sample from the cube')
	# Blob can be defined as a region of an image in which some properties are constant or 
	# vary within a prescribed range of values.
	table = get_intersting_coords(frame, psf_norm, fwhm=fwhm, bkg_sigma=5, plot=opt.plot)
	if opt.show_detections:
		print(table)
	# remove coords having low signal to noise ratio
	snr_thresh = opt.snr #By default is 2
	table = table[table['snr'] > snr_thresh]

	# Plot detection
	if opt.plot :
		plot_detection(frame, table, bounded=False, dpi=100, 
			text_box='We have detected companions from the collapsed (median) frame, and have obtained all possible candidates from the get_interesting_coords function. \
					 However, we have filtered out some of them as they do not show any variation from the background.')

	if opt.fbf:	
    	# How many FWHM we want to consider to fit the model
		cube_final = optimize_params(table, 
								 res_cube, 
								 psf_norm, 
								 fwhm_sphere, 
								 rot_angles, 
								 pixel_scale, 
								 nfwhm=1,
								 method='stddev')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--cube', default='./data/HCI/center_im.fits', type=str,
	                help='Cube file containing coronograph frames sorted by time')
	parser.add_argument('--psf', default='./data/HCI/median_unsat.fits', type=str,
	                help='PSFs file')
	parser.add_argument('--ra', default='./data/HCI/rotnth.fits', type=str,
	                help='Rotational angles')

	parser.add_argument('--w', default=0, type=int,
	                    help='Wavelength to work with')
	parser.add_argument('--p', default=0, type=int,
	                    help='Position of the PSF (init, final) to be used as a reference within the normalization')

	parser.add_argument('--ang_corr', default=0, type=float,
	                    help='Angles correction factor')
	parser.add_argument('--px_corr', default=0.01225, type=float,
	                    help='Pixel scale')
	parser.add_argument('--njobs', default=1, type=int,
	                    help='Number of cores to distribute tasks')
	parser.add_argument('--plot', default=False,
	                    help='Plot every intermidiate step in the pipeline')
	parser.add_argument('--fbf', default=False,
	                    help='True if using frame by frame technique (with optimization). If false only uses the median of the frames without optimization')
	parser.add_argument('--show_detections', default=False,
	                    help='If True prints the positions, flux and snr of possible companions')
	parser.add_argument('--snr', default=2,
	                    help='S/N threshold for deciding whether the blob is a detection or not')
	opt = parser.parse_args()
	run_pipeline(opt)
