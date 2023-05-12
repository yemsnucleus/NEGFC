import numpy as np

from .plottools 						import plot_optimization, plot_mask, plot_to_compare
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

	# from vip_hci.fm.fakecomp import cube_inject_companions
	# frame_fake = cube_inject_companions(frame, 
	# 									psf_norma, 
	# 									ang, 
	# 									-flux, 
	# 									r,
	#                                     plsc=None, 
	#                                     n_branches=1, 
	#                                     theta=theta, imlib='vip-fft',
	#                                     interpolation='lanczos4', transmission=None,
	#                                     radial_gradient=False, full_output=False,
	#                                     verbose=False, nproc=1)

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

	indices = disk((posy, posx), radius=fwhm*2)
	yy, xx = indices
	values = frame_negfc[yy, xx].ravel()    
	
	# plot_to_compare([frame, frame_negfc], ['Frame', 'Frame+FC'])
	# import matplotlib.pyplot as plt
	# plt.figure(figsize=(5,5), dpi=200)
	# plt.imshow(frame_negfc)
	# plt.scatter(posx, posy, marker='x', color='red')
	# plt.scatter(xx, yy, marker='.', s=.5, color='yellow', alpha=0.5)
	# plt.show()
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
        img_shifted = frame_shift(frame_copy, x, y, imlib=imlib)
        tmp += img_shifted*flux
        # plot_to_compare([img_shifted, tmp], ['Shifted PSF', 'Shifted and scaled PSF'])


    frame_out = frame + tmp
    # plot_to_compare([frame, frame_out], ['Frame', 'Frame + FC'])

    return frame_out   