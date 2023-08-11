import multiprocessing as mp
import numpy as np

from vip_hci.fm import cube_inject_companions
from joblib import Parallel, delayed

from core.engine import rotate_cube


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

def get_rings(x, y, fhwm, cube, rot_angles=None, num_rings=None, n_jobs=None):

	if n_jobs is None:
		n_jobs = mp.cpu_count()//2

	num_rings = num_rings+1# since we'll remove the first radius centered in x,y
	radius = fhwm/2
	nframes, width, height = cube.shape
	rad_distances = np.arange(radius, np.ceil(num_rings*radius), radius)

	masks = Parallel(n_jobs=n_jobs)(delayed(create_circle_mask)\
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
	return region

def get_aperture_photometry(cube, x=None, y=None, rot_angles=None, fwhm=4):
	if x is None or y is None:
		x = cube.shape[-1]/2
		y = cube.shape[-2]/2
	
	if rot_angles is not None:
		cube = rotate_cube(cube, rot_angles)

	mask = create_circle_mask(cube.shape[1:-1], (x,y), radius=fwhm/2)
	apflux = np.sum(cube*mask)
	return apflux

def get_contrast(regions, ap_phot, factor=5):
	noises = []
	contrast = []
	factor = 5.
	for r in regions:
		mask_bool = np.where(r!=0)
		noise = np.std(r[mask_bool])
		noises.append(noise)
		contrast.append(factor*noise/ap_phot)
	return contrast

def get_throughput(x_abs, y_abs, cube):
	_, dim0,dim1 = cube.shape

	x = x_abs - dim0/2
	y = y_abs - dim1/2

	radius = np.sqrt((x-dim1/2)**2+(y-dim0/2)**2)
	angle_rad = np.arctan2((y-dim1/2), (x-dim0/2)) 
	angle_degree = angle_rad*180/np.pi+360

	print(angle_degree)