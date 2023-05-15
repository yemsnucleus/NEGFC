import pandas as pd
import numpy as np 
import time
import os

from vip_hci.fits import open_fits


def get_data(root='./data/HCI', lambda_ch=0, psf_pos=0):
	cube_route = os.path.join(root, 'center_im.fits')
	cube       = open_fits(cube_route, header=False) 

	psf_route = os.path.join(root, 'median_unsat.fits')
	psf       = open_fits(psf_route, header=False)

	ra_route = os.path.join(root, 'rotnth.fits')
	rot_ang  = open_fits(ra_route, header=False)
	rot_ang  = -rot_ang

	return {'cube': cube, 
			'psf':ra_route,
			'rot_angles': rot_ang}

def preprocess_psf(cube):

	if len(cube[..., 0])%2 == 0:
		ycen=ycube_center-0.5
		xcen=xcube_center-0.5
		newdim=len(cube[1,:,1])-1


ycen=ycube_center-0.5
xcen=xcube_center-0.5
newdim=len(cube[1,:,1])-1
cube_crop=cube_crop_frames(cube_shifted,newdim,xy=[int(ycen),int(xcen)], force=True)  

def run():

	t0 = time.time()
	data = get_data('./data/HCI')
	t1 = time.time()






run()