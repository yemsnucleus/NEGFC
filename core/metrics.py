from scipy import stats
import multiprocessing as mp
import pandas as pd
import numpy as np
import vip_hci as vip

from vip_hci.fm import cube_inject_companions
from joblib import Parallel, delayed

from core.engine import rotate_cube, preprocess, first_guess, get_angle_radius


def inject_companion(x, y, flux, cube, psf, rot_angles, pixel=0.01225):

	cube_center = cube.shape[-1]/2

	radius = np.sqrt((x-cube_center)**2+(y-cube_center)**2)
	angle_rad = np.arctan2((y-cube_center), (x-cube_center)) 
	angle_degree = angle_rad*180/np.pi+360

	fakecomp = cube_inject_companions(cube, 
	                                  psf, 
	                                  rot_angles, 
	                                  flux, 
	                                  pixel, 
	                                  radius, 
	                                  theta=angle_degree)
	return fakecomp

def create_circle_mask(image_shape, center, radius, n_jobs=4):
	rows, cols = image_shape
	Y, X = np.ogrid[:rows, :cols]
	distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
	mask = distance_from_center <= radius
	return mask

def get_rings(x, y, fhwm, cube, rot_angles=None, num_rings=None, n_jobs=1):

	if n_jobs is None:
		n_jobs = mp.cpu_count()//2

	num_rings = num_rings+1# since we'll remove the first radius centered in x,y
	radius = fhwm/2
	nframes, width, height = cube.shape
	rad_distances = np.arange(radius, np.ceil(num_rings*radius), radius)

	masks = Parallel(n_jobs=1)(delayed(create_circle_mask)\
							((width, height), 
							 [x, y], 
							 rdist) \
							for rdist in rad_distances)
	masks = np.array(masks)
	masks = np.logical_and(masks[1:], np.logical_not(masks[0:-1]))
	masks = masks.astype(float)

	if rot_angles is not None:
		cube = rotate_cube(cube, rot_angles)

	if cube.shape[-1] == 1:
		cube = np.squeeze(cube, axis=-1)

	region = cube[None,...]*masks[:, None,...]
	return region, rad_distances[:-1]

def get_aperture_photometry(cube, fwhm=4):
	x = cube.shape[-1]/2
	y = cube.shape[-2]/2
	mask = create_circle_mask(cube.shape[1:], (x,y), radius=fwhm/2)
	cube_masked = cube*mask[None,...]
	apflux = np.sum(cube_masked, axis=(1, 2))
	apflux = np.mean(apflux)	
	return apflux

def get_contrast(regions, ap_phot, factor=5):
	noises = []
	contrast = []
	factor = 5.
	for r in regions:
		mask_bool = np.where(r!=0)
		in_region = r[mask_bool]
		noise = np.std(r[mask_bool])
		noises.append(noise)
		contrast.append(factor*noise/ap_phot)
	return contrast

def find_parameters(table, cube, psf, rot_angles, window_size):
	inputs = {'cube': cube, 'psf':psf, 'rot_angles':rot_angles}
	_, cube, psf, rot_angles, backmoments = preprocess(inputs)

	table = first_guess(table, cube, psf, 
				backmoments=backmoments,
				window_size=window_size, 
				learning_rate=1e-0, 
				epochs=1e6,
				target_folder=None,
				verbose=0,
				loss_precision=0.)
	return table

def vip_firstguess(table, cube, psf, rot_angles, window_size):
	print("Data is being loaded as in function 'find_parameters'")
	inputs = {'cube': cube, 'psf':psf, 'rot_angles':rot_angles}
	_, cube, psf, rot_angles, backmoments = preprocess(inputs)
	#el cube esta derotated pero hay que obtener el fwhm del psf y normalizar el psf pa pasarlo
	#falta el centro de la imagen y checkear la transformacion
          
	optimal_fluxes, optimal_xs, optimal_ys = [], [], []
	nfwhm = 3
	print("PSF SHAPE", psf.shape)
	for i in range(len(table)):
		r, theta, f  = vip.fm.negfc_simplex.firstguess(cube, 
											rot_angles,       
											psf[0],
											fmerit = 'stddev',
											ncomp=1, 
											planets_xy_coord=[(table.iloc[i].x, table.iloc[i].y)],
											fwhm=int(nfwhm)*float(table.iloc[i].fwhm_mean),
											simplex=True,
											verbose=False,
											annulus_width=int(nfwhm*(table.iloc[i].fwhm_mean)),
											aperture_radius=2,
											f_range=np.linspace(table.iloc[i].flux -10, table.iloc[i].flux*2, 10),
											imlib='opencv',
											interpolation='lanczos4',
											plot=False,
											mu_sigma=False)
		posy = r * np.sin(np.deg2rad(theta)) + np.ceil(cube.shape[1]/2)
		posx = r * np.cos(np.deg2rad(theta)) + np.ceil(cube.shape[2]/2)
		optimal_fluxes.append(f)
		optimal_xs.append(posx)
		optimal_ys.append(posy)
	table['optimal_flux'] = optimal_fluxes
	table['optimal_x'] = optimal_xs
	table['optimal_y'] = optimal_ys 
	return table
          

def get_throughput(cube, psf, rot_angles, fwhm, rad_distances, regions, K=4, window_size=15, method="tf", pixel=0.01225, n_jobs=None, optimize=True):
	""" Throughput function that measure the performance of the processing image algorithms
	
	We consider the following pipeline: 
		- For each ring inject K fake companions around the center
		- Estimates the flux and optimal positions
		- Remove the companion from the image
		- Calculate the ratio between residuals/injected

	Args:
		cube (numpy.array): Clean cube (with real companions removed)
		psf (numpy.array): Normalized 1FWHM PSFs
		rot_angles (list): Rotation angles
		flux ([type]): Flux
		fwhm ([type]): [description]
		window_size (number): [description] (default: `15`)
		pixel (number): [description] (default: `0.01225`)
	
	Returns:
		[type]: [description]
	"""
	if n_jobs is None:
		n_jobs = mp.cpu_count()//2

	_, dim0, dim1 = cube.shape
	
	theta_angles = np.linspace(0, 360, K+1) [1:]
	fluxes = np.random.uniform(cube.std()*10, cube.std()*20, K)

	throughputs = []
	tables = []
	for radius in rad_distances:
		fake_comps = Parallel(n_jobs=n_jobs)(delayed(cube_inject_companions)\
											(np.zeros_like(cube), psf[0], rot_angles, fluxes[i], pixel, radius, 1, theta_angles[i]) \
											for i in range(len(theta_angles)))
		fake_comps = np.sum(fake_comps, axis=0)
		injected_cube = fake_comps + cube


		posy = radius * np.sin(np.deg2rad(theta_angles)) + dim1/2. - 0.5
		posx = radius * np.cos(np.deg2rad(theta_angles)) + dim0/2. - 0.5


		table = pd.DataFrame({'x': posx, 'y': posy, 'flux': fluxes-(cube.std()*3), 'fwhm_mean':fwhm})
		if optimize:
			if method == "vip":
				table = vip_firstguess(table, injected_cube, psf, rot_angles, window_size)
			elif method == "tf":
				table = find_parameters(table, injected_cube, psf, rot_angles, window_size)
		else:
			table['optimal_flux'] = fluxes
			table['optimal_x'] = posx
			table['optimal_y'] = posy

		tables.append(table)

		r_star, theta_star = get_angle_radius(table['optimal_x'], table['optimal_y'], 
											  width=dim1, height=dim0)

		fakecomp_pred = cube_inject_companions(np.zeros_like(cube), 
			                                   psf[0], 
			                                   rot_angles, 
			                                   table['optimal_flux'].values[0], 
			                                   pixel, 
			                                   r_star.values[0], 
			                                   theta=theta_star.values[0])

		residuals = injected_cube - fakecomp_pred

		tput = residuals/injected_cube
		tput = rotate_cube(tput, rot_angles)
		mask = np.where(regions!=0., 1., 0.)
		valid_region = tput[...,0]*mask[1]
		throughput = np.mean(valid_region)
		throughputs.append(throughput)

	tables = pd.concat(tables)
	tables['rad_dist'] = np.repeat(rad_distances, K)
	return throughputs, tables


def correct_contrast(contrast_curve, regions, ap_phot):
	corrected_contrasts = []
	for contrast, region in zip(contrast_curve, regions):
		in_region = region[region!=0]
		df_2 = in_region.shape[0]-1
		tau = stats.t.ppf(q=contrast, df=df_2)
		stddev = np.std(in_region)

		corr_contrast = -tau*stddev*np.sqrt(1+1/(df_2))/ap_phot 	
		corrected_contrasts.append(corr_contrast)
	return corrected_contrasts

# from photutils.aperture import aperture_photometry, CircularAperture
# aper = CircularAperture((psfs.shape[1]/2, psfs.shape[2]/2), 
#                         r=row['fwhm_mean']) 

# obj_flux_i = aperture_photometry(psfs[0], aper, method='exact')
# ap_phot = obj_flux_i['aperture_sum'][0]
# ap_phot