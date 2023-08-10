from vip_hci.fm import cube_inject_companions
import numpy as np


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