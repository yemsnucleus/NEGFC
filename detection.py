import pandas as pd
import numpy as np

from astropy.stats 					import sigma_clipped_stats, gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling				import models, fitting
from plottools 						import plot_to_compare
from skimage.feature				import peak_local_max
from vip_hci.preproc.recentering 	import frame_center
from vip_hci.var.shapes				import get_square
from scipy.ndimage.filters 			import correlate
from vip_hci.metrics.snr_source		import snr
from loss	 						import chisquare_mod
from scipy.optimize					import minimize

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

	# Padding the image with zeros to avoid errors at the edges
	pad_value = 10
	array_padded = np.pad(frame, pad_width=pad_value, mode='constant', constant_values=0)
	if plot:
		plot_to_compare([frame, array_padded], ['Original', 'Padded'])

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

def optimize_params(table, cube, psf, fwhm, rot_angles, pixel_scale, nfwhm=3, plot=False):
	
	fwhma = int(nfwhm)*float(fwhm)
	
	n_frames, height, width = cube.shape
	# Cube to store the final model
	cube_emp= np.zeros_like(cube)
	f_0_comp, r_0_comp,  theta_0_comp= np.zeros(int(n_frames)), \
									   np.zeros(int(n_frames)), \
									   np.zeros(int(n_frames))
	
	x_cube_center, y_cube_center = frame_center(cube[0])

	if plot:
		plot_detection(frame, table)

	print(':'*100)
	for _, row in table.iterrows():
		x 	 = float(row['x']) - x_cube_center
		y    = float(row['y']) - y_cube_center
		flux = row['flux']
		print('[INFO] Optimizing companion located at ({:.2f}, {:.2f}) with flux {:.2f}'.format(x, y, flux))
		for index in range(n_frames): 
			current_frame = cube[index]

			radius = np.sqrt(x**2+y**2) # radius
			angle  = np.arctan2(y, x)   # radians
			angle  = angle/np.pi*180    # degrees
			# Since atan2 return angles in between [0, 180] and [-180, 0],
			# we convert the angle to refers a system of 360 degrees
			theta0 = np.mod(angle, 360) 
			params = (radius, theta0, flux)

			solu = minimize(chisquare_mod, params, 
				args=(row['x'], row['y'], 
					  current_frame, 
					  -rot_angles[index], 
					  pixel_scale, 
					  psf, 
					  fwhma, 
					  'stddev'),
    			method = 'Nelder-Mead')

			# radius, theta0, flux = solu.x
			# f_0_comp[index]=flux
			# r_0_comp[index]=radius
			# theta_0_comp[index]=theta0

			# frame_emp = inject_fcs_cube_mod(current_frame, 
			# 							  	psf_norm, 
			# 							  	-rot_angles[index], 
			# 							  	-flux, 
			# 							  	radius, 
			# 							  	theta0, 
			# 							  	n_branches=1)
			# cube_emp[index,:,:]=frame_emp