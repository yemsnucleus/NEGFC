import matplotlib.pyplot as plt
import numpy as np

from joblib import Parallel, delayed

from vip_hci.preproc.recentering import frame_shift, frame_center, cube_recenter_2dfit
from vip_hci.preproc.cosmetics 	 import cube_crop_frames
from vip_hci.var 				 import fit_2dgaussian
from vip_hci.fm 				 import normalize_psf
from vip_hci.var.shapes			 import get_square
from vip_hci.fits 				 import open_fits

# Factor with which to multiply Gaussian FWHM to convert it to 1-sigma standard deviation
from astropy.stats 				 import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling 			 import models, fitting

from photutils.centroids 		 import centroid_com


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

def plot_to_compare(images, titles, axes=None):
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
		fig, axes = plt.subplots(1, len(images))
	for i, (im, ti) in enumerate(zip(images, titles)):
		axes[i].imshow(im)
		axes[i].set_title(ti)
	plt.show()
	return axes 

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


def run_pipeline(cube_path, psf_path, rot_ang_path, wavelength=0, psf_pos=0, pixel_scale=0.01225, n_jobs=1):
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
	rot_angles = open_fits(rot_ang_path, header=False) # didn't use. Should we use it when defininf the gaussian?

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

	plot_to_compare([single_psf, psf_subimage], ['Original', 'Subimage'])
	
	fwhm_y, fwhm_x, mean_y, mean_x = fit_gaussian_2d(psf_subimage, plot=True)
	mean_y +=  suby # put the subimage in the original center
	mean_x +=  subx # put the subimage in the original center

	fwhm_sphere  = np.mean([fwhm_y, fwhm_x]) # Shared across the frames 
	psf_rec = recenter_cube(psf[wavelength], single_psf, fwhm_sphere=fwhm_sphere, n_jobs=n_jobs)
	
	# Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM aperture equal to one. 
	# It also allows to crop the array and center the PSF at the center of the array(s).
	psf_norm, fwhm_flux,fwhm = normalize_psf(psf_rec[psf_pos], 
	                                         fwhm=fwhm_sphere,
	                                         size=None, 
	                                         threshold=None, 
	                                         mask_core=None,
	                                         full_output=True, 
	                                         verbose=False) 
	plot_to_compare([psf_rec[psf_pos], psf_norm], ['PSF reconstructed', 'PSF normalized'])





if __name__ == '__main__':

	# # Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM aperture equal to one. 
	# # It also allows to crop the array and center the PSF at the center of the array(s).
	# # QUESTIONS:
	# # 1) Can I normalice both PSFs using the same fitted model? is it worth?
	# # 2) Can I get FWHM directly from the pixel distribution WITHOUT FITTING A GAUSSIAN?

	cube_path    = './data/HCI/center_im.fits'
	psf_path     = './data/HCI/median_unsat.fits' # Why this name?
	rot_ang_path = './data/HCI/rotnth.fits'

	run_pipeline(cube_path, psf_path, rot_ang_path, n_jobs=2)