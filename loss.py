import numpy as np

from plottools 						import plot_optimization, plot_mask
from vip_hci.preproc.recentering	import frame_shift, frame_center
from skimage.draw					import disk

def chisquare_mod(modelParameters, frame, ang, pixel, psf_norma, fwhm, fmerit):
	"""Creates the objetive function to be minimized

	This function calculate residuals (errors) based on first guesses 
	of the physical parameters of the companion candidate.
	:param modelParameters: Parameters to be adjusted
	:type modelParameters: list of scalars
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

	indices = disk((posy, posx), radius=fwhm)
	yy, xx = indices
	values = frame_negfc[yy, xx].ravel()    
	
	# plot_mask(frame_negfc, posx, posy, fwhm)

	# Function of merit
	if fmerit == 'sum':
		values = np.abs(values)
		chi2 = np.sum(values[values > 0])
		N = len(values[values > 0])
		loss =  chi2 / (N-3) 
	if fmerit == 'stddev':
	    loss = np.std(values[values != 0]) # loss

	# plot_optimization(frame, frame_negfc, 
	# 				  msg='Residuals {:.2f}'.format(loss), 
	# 				  root='./figures/negfc_opt/')
	return loss

def inject_fcs_cube_mod(frame, template, angle, flux, radius, theta, n_branches=1, imlib='opencv'):
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
		import matplotlib.pyplot as plt
		tmp += img_shifted*flux
	frame_out = frame + tmp

	return frame_out   