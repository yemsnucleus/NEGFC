import pandas as pd
import numpy as np

from astropy.stats 					import sigma_clipped_stats, gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from loss	 						import chisquare_mod, inject_fcs_cube_mod
from plottools 						import plot_to_compare, plot_detection
from astropy.modeling				import models, fitting
from skimage.feature				import peak_local_max
from vip_hci.preproc.recentering 	import frame_center
from vip_hci.var.shapes				import get_square
from scipy.ndimage.filters 			import correlate
from scipy.optimize					import minimize
from vip_hci.metrics.snr_source		import snr
from tqdm 							import tqdm

def dist(yc, xc, y1, x1): #function from vip_hci
    """
    Return the Euclidean distance between two points, or between an array
    of positions and a point.
    """
    return np.sqrt(np.power(yc-y1, 2) + np.power(xc-x1, 2))

def get_intersting_coords(frame, psf_norm, fwhm=4, bkg_sigma = 5, plot=False):
	"""Get coordinates of potential companions
	"""
	#Calculate sigma-clipped statistics on the provided data.
	_, median, stddev = sigma_clipped_stats(frame, sigma=bkg_sigma, maxiters=None)
	bkg_level = median + (stddev * bkg_sigma)

	# Padding the image with zeros to avoid errors at the edges
	pad_value = 10
    array_padded = torch.nn.functional.pad(frame, (pad_value, pad_value), mode='constant', constant_values=0)
	
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
		if fit.amplitude.value > 0 and condxf and condyf and condmf:
			coords.append((suby + fit.y_mean.value,
						   subx + fit.x_mean.value))
			fluxes.append(fit.amplitude.value)
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
	return table

