from vip_hci.preproc.recentering import frame_shift, frame_center, cube_recenter_2dfit
from vip_hci.fm import cube_inject_companions, cube_planet_free, firstguess, normalize_psf
from vip_hci.var import fit_2dgaussian, get_square, get_annulus_segments
from vip_hci.preproc.derotation import cube_derotate
from vip_hci.metrics import aperture_flux
from vip_hci.psfsub import median_sub

from core.engine import get_angle_radius, preprocess
from scipy.stats import t as tstudent

import matplotlib.pyplot as plt 
from astropy.io import fits
import pandas as pd
import numpy as np
import os

def chisquare_mod ( modelParameters,sourcex, sourcey, frame, ang, plsc, psf_norma, fwhm, fmerit,
			  svd_mode='lapack'):
		try:
			r, theta, flux = modelParameters
		except TypeError:
			print('paraVector must be a tuple, {} was given'.format(type(modelParameters)))
		frame_negfc = inject_fcs_cube_mod(frame, psf_norma, ang, -flux, pixel, r, theta,n_branches=1)
		centy_fr, centx_fr = frame_center(frame_negfc)
		posy = r * np.sin(np.deg2rad(theta)-np.deg2rad(ang)) + centy_fr
		posx = r * np.cos(np.deg2rad(theta)-np.deg2rad(ang)) + centx_fr
		indices = circle(posy, posx, radius=2*fwhm)
		yy, xx = indices
		values = frame_negfc[yy, xx].ravel()    
		# Function of merit
		if fmerit == 'sum':
			values = np.abs(values)
			chi2 = np.sum(values[values > 0])
			N = len(values[values > 0])
			return chi2 / (N-3)
		elif fmerit == 'stddev':
			return np.std(values[values != 0])
		else:
			raise RuntimeError('`fmerit` choice not recognized')

def inject_fcs_cube_mod(array, psf_template, angle_list, flevel, plsc, rad_dists, 
					theta,n_branches=1, imlib='opencv', verbose=True):
	ceny, cenx = frame_center(array)
	size_fc = psf_template.shape[0]
	fc_fr = np.zeros_like(array, dtype=np.float64)  # TODO: why float64?
	w = int(np.floor(size_fc/2.))
	# fcomp in the center of a zeros frame
	fc_fr[int(ceny-w):int(ceny+w+1), int(cenx-w):int(cenx+w+1)] = psf_template

	array_out = np.zeros_like(array)
	tmp = np.zeros_like(array)
	for branch in range(n_branches):
		ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta)
		rad = rad_dists
		y = rad * np.sin(ang - np.deg2rad(angle_list))
		x = rad * np.cos(ang - np.deg2rad(angle_list))
		tmp += frame_shift(fc_fr, y, x, imlib=imlib)*flevel
	array_out = array + tmp
		
	return array_out   

def center_normalize_psf(psf, cropsize=30):
	fit = fit_2dgaussian(psf[0], 
						 crop=True, 
						 cropsize=cropsize, 
						 debug=False, 
						 full_output=True)  
	mean_fwhm_fit = np.mean([fit.fwhm_y, fit.fwhm_x])

	y_cent, x_cent = frame_center(psf[0]) 
	y_c=int(y_cent)
	x_c=int(x_cent)
	psf_center, y0_sh, x0_sh = cube_recenter_2dfit(psf, 
												    (y_c, x_c), 
												    mean_fwhm_fit,
													model='gauss',
													nproc=1, 
													subi_size=7, 
													negative=False,
													full_output=False, 
													debug=False) 

	psf_norm, fwhm_flux, fwhm = normalize_psf(psf_center, 
	                                         fwhm=mean_fwhm_fit, 
	                                         full_output=True, 
	                                         verbose=False)

	return psf_norm, fwhm_flux, fwhm


def load_folder(data_folder):
	cube   = fits.getdata(os.path.join(data_folder, 'center_im.fits'), ext=0)
	psfs   = fits.getdata(os.path.join(data_folder, 'median_unsat.fits'), ext=0)
	angles = fits.getdata(os.path.join(data_folder, 'rotnth.fits'), ext=0)
	angles = -angles
	return cube, psfs, angles

def get_clean_cube(cube, psfs, angles, params_table, pixel=0.01225):
	psf_norm, ap_flux, fwhm = center_normalize_psf(psfs, cropsize=30)

	comp_params = []
	crop_psfs = []
	for index, row in params_table.iterrows():
		# Angular differential imaging
		radius, theta = get_angle_radius(row['optimal_x'], 
										 row['optimal_y'], 
										 width=cube.shape[-1],
										 height=cube.shape[-2])
		comp_params.append((radius, theta, row['optimal_flux']))

		new_psf_size = 3 * round(row['fwhm_mean'])
		if new_psf_size % 2 == 0: new_psf_size += 1
		psf_norm_crop = normalize_psf(psf_norm, 
									  fwhm=row['fwhm_mean'],
									  size=min(new_psf_size, psf_norm.shape[-1]))
		crop_psfs.append(psf_norm_crop)
	
	cube_emp = cube_planet_free(comp_params, cube, angles, psf_norm, pixel)

	adi_cube  = median_sub(cube, angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')
	adi_empty = median_sub(cube_emp, angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')
	
	return adi_cube, adi_empty, crop_psfs


def get_xy_from_angle(angle, radius):
	if 0<=angle<=90:
		x=radius/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
		y=radius*np.tan(angle*np.pi/180.)/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
	if 90<angle<=180:
		x=-radius/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
		y=-radius*np.tan(angle*np.pi/180.)/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
	if 180<angle<=270:
		x=-radius/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
		y=-radius*np.tan(angle*np.pi/180.)/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
	if 270<angle<=360:
		x=radius/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
		y=radius*np.tan(angle*np.pi/180.)/np.sqrt(1+(np.tan(angle*np.pi/180.))**2)
	return x, y

def get_fakecomp_to_inject(residuals_frame, params_table, offset=9):
	x_fake_positions    = []
	y_fake_positions    = []
	flux_fake_positions = []
	frame_squares = []
	rad_distances = []
	for index, row in params_table.iterrows():
		# why offset*2+1 ?
		frame_sq = get_square(residuals_frame, offset*2+1, row['optimal_y'], row['optimal_x'])
		fwhm = round(row['fwhm_mean'])
		sq_noises = []
		rad_dists = []
		# why offset + 2?
		for i in range(fwhm, offset+2, fwhm):
			annulus = get_annulus_segments(frame_sq, i, fwhm, mode='val')[0]
			stdev   = np.std(annulus[np.where(annulus !=0.)])
			rad_dists.append(i)
			sq_noises.append(stdev)
		rad_distances.append(rad_dists)

		x_fake_comp = []
		y_fake_comp = []
		flux_fake_comp = []
		end_dist  = int(np.max(rad_dists))
		for i, r in enumerate(range(fwhm, end_dist, fwhm)):
			rad_angle = np.arctan2(fwhm, r) 
			step = int(rad_angle*180/np.pi) # degrees

			for angle in range(0, 360, step):
				x, y = get_xy_from_angle(angle, r)
				flux=sq_noises[i]*10.
				x_fake_comp.append(x)
				y_fake_comp.append(y)
				flux_fake_comp.append(flux)

		frame_squares.append(frame_sq)
		x_fake_positions.append(np.asarray(x_fake_comp))
		y_fake_positions.append(np.asarray(y_fake_comp))
		flux_fake_positions.append(np.asarray(flux_fake_comp))

	return x_fake_positions, y_fake_positions, flux_fake_positions, frame_squares, rad_distances

def get_throughput(xs_fake, ys_fake, fluxes_fake, psf, cube, adi_cube_res, rot_angles, params_table, pixel=0.01225):
	# Should we inject on the oriignal cube?
	params_table = params_table.reset_index(drop=True)
	
	throughput_list = []
	for index, row in params_table.iterrows():
		print(f'[INFO] Processing row {index}')
		throughput = []
		cube_constback = np.ones_like(cube)*1e-6

		xp, yp = row['optimal_x'], row['optimal_y']
		cube_center = cube.shape[-1]/2

		# for k in range(len(xfc)):
		for xpos, ypos, flux in zip(xs_fake[index], ys_fake[index], fluxes_fake[index]): 
			print(f'[INFO] Processing Fake (x/y)=({xpos},{ypos}) and flux: {flux}')
			x_abs = xp + xpos
			y_abs = yp + ypos

			# distance between cube center and fake companion
			dist_center_fc = np.sqrt((x_abs-cube_center)**2+(y_abs-cube_center)**2)
			angle_rad = np.arctan2((y_abs-cube_center), (x_abs-cube_center)) 
			angle_degree = angle_rad*180/np.pi+360

			fakecomp = cube_inject_companions(cube, psf, rot_angles, flux, pixel, dist_center_fc, theta=angle_degree)

			r_0, theta_0, f_0 = firstguess(fakecomp, 
										   rot_angles, 
										   psf, 
										   annulus_width=1, 
										   aperture_radius=1,
										   ncomp=1,
										   plsc=pixel, 
										   fmerit='stddev',
										   planets_xy_coord=[(xp, yp)], 
										   simplex=False,
										   fwhm=row['fwhm_mean'],
										   imlib='opencv', 
										   interpolation='nearneig')

			plpar = [(r_0[0], theta_0[0], f_0[0])]
			cube_fake_removed  = cube_planet_free(plpar, fakecomp, rot_angles, psf, pixel, imlib='opencv',interpolation='lanczos4')
			frame_fake_removed = median_sub(cube_fake_removed, rot_angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')

			cube_planet_fc_map = cube_inject_companions(cube_constback,
														psf,
														rot_angles,
														flux,
														pixel,
														dist_center_fc,
														theta=angle_degree)

			frame_fake = median_sub(cube_planet_fc_map, rot_angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')


			injected_flux  = aperture_flux(frame_fake, [y_abs], [x_abs], row['fwhm_mean'], ap_factor=1, mean=False)
			recovered_flux = aperture_flux((frame_fake_removed-adi_cube_res), [y_abs], [x_abs], row['fwhm_mean'], ap_factor=1, mean=False)
			
			throughput_val = recovered_flux[0] / injected_flux[0]
			throughput.append([xpos, ypos, throughput_val])
		throughput_list.append(throughput)
	return throughput_list


def post_processing_throughput(throughput_list, rad_distances, table, fr_adi_res, psf, offset=9):
	cont_list, dist_list = [], []
	for throughput, rad_dist, (index, row) in zip(throughput_list, rad_distances, table.iterrows()):
		throughput = np.asarray(throughput)
		rad_dist   = np.asarray(rad_dist)
		i=0
		j=0
		throughput_sum=0
		throughput_mean=[]
		somma=0
		while i<=len(throughput):
			if i==len(throughput):
				throughput_mean.append([throughput_sum/somma,rad_dist[j]])
			else :
				xpos  = throughput[i,0]
				ypos  = throughput[i,1]
				thput = throughput[i,2]
				
				r=np.sqrt(xpos**2+ypos**2)
				if r<=rad_dist[j]:
					throughput_sum+=thput
					somma=somma+1
				else:
					throughput_mean.append([throughput_sum/somma, rad_dist[j]])
					throughput_sum=0
					somma=0
					throughput_sum+=thput
					somma=somma+1
					j=j+1
			i=i+1

		throughput_mean=np.asarray(throughput_mean)
		res=fr_adi_res[(int(row['optimal_y'])-offset):(int(row['optimal_y'])+offset),
					   (int(row['optimal_x'])-offset):(int(row['optimal_x'])+offset)]

		r=row['fwhm_mean']/2.
		cont, dist = [], []
		j=0
		while r<offset:
			ann = get_annulus_segments(res, r, r, mode='val')
			ann_abs=np.abs(ann)
			mean=np.mean(ann_abs[ann_abs != 0])
			stddev=np.std(ann_abs[ann_abs != 0])

			contrast=5*stddev/np.max(psf)
			tcorr=tstudent.ppf(contrast,2*np.pi*r-1)
			tcorr=-tcorr
			contrastcorr=(tcorr*stddev*np.sqrt(1+1/(2*np.pi*r))+mean)/np.max(psf)

			thput_soma = throughput_mean[j,0]
			raddist    = throughput_mean[j,1]
			if (r<=raddist) or ((j+1)>=len(throughput_mean[:,0])):
				contrastcorr_trp=contrastcorr*(1/thput_soma)
			else :
				j=j+1
				contrastcorr_trp=contrastcorr*(1/thput_soma)

			dist.append(r)
			cont.append(contrastcorr_trp)
			r=r+1

		dist_list.append(dist)
		cont_list.append(cont)

	return dist_list, cont_list






# out_file.close()