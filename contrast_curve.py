from vip_hci.fm import cube_inject_companions, cube_planet_free
from vip_hci.var import get_square, get_annulus_segments
from vip_hci.preproc.derotation import cube_derotate
from vip_hci.psfsub import median_sub
from vip_hci.fm import normalize_psf, firstguess
from vip_hci.metrics import aperture_flux
from scipy.stats import t as tstudent
from core.engine import get_angle_radius, preprocess

import matplotlib.pyplot as plt 
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
	
path 	  = './logs/f_dhtau/'
data_path = './data/real/f_dhtau'
pixel     = 0.01225

table, cube, psf, rot_angles, backmoments = preprocess(data_path)
rot_angles = -rot_angles
cube = cube_derotate(cube, rot_angles, nproc=4, imlib='opencv', interpolation='nearneig')
psf_norm = psf[0]

comp_parameters = pd.read_csv(os.path.join(path, 'prediction.csv'))
row = comp_parameters.iloc[0]

# Angular differential imaging
radius, theta = get_angle_radius(row['optimal_x'], 
								 row['optimal_y'], 
								 width=cube.shape[-1],
								 height=cube.shape[-2])

cube_emp   = cube_planet_free([(radius, theta, row['optimal_flux'])], cube, rot_angles, psf_norm, pixel)
fr_adi_res = median_sub(cube_emp, rot_angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')

offset=9 # why this?
frame_square = get_square(fr_adi_res, offset*2+1, row['optimal_y'], row['optimal_x'])

# ====== Get noise at different radial distances ======
rad_dist=[]
noise=[]
# why offset + 2?
fwhm_round = int(round(row['fwhm_mean']))

for i in range(fwhm_round, int(offset+2), fwhm_round):
	annulus_indices = get_annulus_segments(frame_square, i, round(row['fwhm_mean']))
	annulus = frame_square[annulus_indices]
	stdev   = np.std(annulus[np.where(annulus !=0.)])
	rad_dist.append(i)
	noise.append(stdev)

new_psf_size = 3 * fwhm_round
if new_psf_size % 2 == 0:
	new_psf_size += 1

if cube.ndim == 3:
	psf_norm_crop = normalize_psf(psf_norm, 
								  fwhm=row['fwhm_mean'],
								  size=min(new_psf_size, psf_norm.shape[1]))

# ========= I DONT KNOW WHAT IS HAPPENING HERE =========
xfc=[]
yfc=[]
ffc=[]
i=0


fwhm_0 = fwhm_round
fwhm_1 = int(np.max(rad_dist))
for r in range(fwhm_0, fwhm_1, ):
	y = round(row['fwhm_mean'])
	x = r
	step = int(np.arctan(/r)*180/np.pi)
	
	for theta in range(0, 360, step):
		if 0<=theta<=90:
			x=r/np.sqrt(1+(np.ta2n(theta*np.pi/180.))**2)
			y=r*np.tan2(theta*np.pi/180.)/np.sqrt(1+(np.tan2(theta*np.pi/180.))**2)
		elif 90<theta<=180:
			x=-r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
			y=-r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
		elif 180<theta<=270:
			x=-r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
			y=-r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
		else :
			x=r/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)
			y=r*np.tan(theta*np.pi/180.)/np.sqrt(1+(np.tan(theta*np.pi/180.))**2)

		flux=noise[i]*10.
		xfc.append(x)
		yfc.append(y)
		ffc.append(flux)
	i=i+1
xfc=np.asarray(xfc)
yfc=np.asarray(yfc)
ffc=np.asarray(ffc)
plt.figure()
plt.plot(xfc, yfc)
plt.savefig('./figures/curve_partial.png')

# ==========
throughput=[]
cube_crop2     = np.zeros_like(cube)
cube_crop3     = np.ones_like(cube)*1e-6
cube_planet_fc = np.zeros_like(cube)
cube_emp_pl    = np.zeros_like(cube)
xp, yp = row['optimal_x'], row['optimal_y']
cube_width = cube.shape[-1]/2
source_xy = [(xp, yp)] 
flx_min = row['optimal_flux'] - 5
flx_max = row['optimal_flux'] + 5
for k in range(len(xfc)):
	cube_crop2=cube
	rfc=np.sqrt((xfc[k]+xp-cube_width)**2+(yfc[k]+yp-cube_width)**2)
	if (xfc[k]+xp)>=cube_width and (yfc[k]+yp)>=cube_width:
		thetafc=np.arctan((yfc[k]+yp-cube_width)/(xfc[k]+xp-cube_width))*180/np.pi
	elif ((xfc[k]+xp)<cube_width and (yfc[k]+yp)>cube_width) or ((xfc[k]+xp)<cube_width and (yfc[k]+yp)<cube_width):
		thetafc=np.arctan((yfc[k]+yp-cube_width)/(xfc[k]+xp-cube_width))*180/np.pi+180
	else:
		thetafc=np.arctan((yfc[k]+yp-cube_width)/(xfc[k]+xp-cube_width))*180/np.pi+360

	cube_planet_fc=cube_inject_companions(cube_crop2, psf_norm_crop, rot_angles, ffc[k], pixel, rfc, theta=thetafc)

	r_0, theta_0, f_0 = firstguess(cube_planet_fc, rot_angles, psf_norm, annulus_width=1, aperture_radius=1,
									ncomp=1,
									plsc=pixel, 
									fmerit='stddev',
									planets_xy_coord=source_xy, 
									simplex=False,
									fwhm=row['fwhm_mean'],
									imlib='opencv', 
									interpolation='nearneig',
									f_range=np.linspace(flx_min,flx_max,10))
	plpar = [(r_0[0], theta_0[0], f_0[0])]
	cube_emp_pl=cube_planet_free(plpar, cube_planet_fc, rot_angles, psf_norm, pixel, imlib='opencv',interpolation='lanczos4')

	fr_adi_fc = median_sub(cube_emp_pl, rot_angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')

	cube_planet_fc_map=cube_inject_companions(cube_crop3,
											  psf_norm_crop,
											  rot_angles,ffc[k],
											  pixel,
											  rfc,
											  theta=thetafc)

	fr_adi_map = median_sub(cube_planet_fc_map, rot_angles, imlib='opencv', interpolation='lanczos4', mode='fullfr')

	injected_flux = aperture_flux(fr_adi_map, [(yfc[k]+yp)], [(xfc[k]+xp)], row['fwhm_mean'],
											  ap_factor=1, mean=False)
	recovered_flux = aperture_flux((fr_adi_fc-fr_adi_res), [(yfc[k]+yp)],
											   [(xfc[k]+xp)], row['fwhm_mean'], ap_factor=1,
											   mean=False)
	thruput = recovered_flux[0] / injected_flux[0]
	throughput.append([xfc[k],yfc[k],thruput])

throughput=np.asarray(throughput)
rad_dist=np.asarray(rad_dist)
i=0
j=0
throughput_sum=0
throughput_mean=[]
somma=0
while i<=len(throughput):
	if i==len(throughput):
		throughput_mean.append([throughput_sum/somma,rad_dist[j]])
	else :
		r=np.sqrt((throughput[i,0])**2+(throughput[i,1])**2)
		if r<=rad_dist[j]:
			throughput_sum=throughput_sum+throughput[i,2]
			somma=somma+1
		else:
			throughput_mean.append([throughput_sum/somma,rad_dist[j]])
			throughput_sum=0
			somma=0
			throughput_sum=throughput_sum+throughput[i,2]
			somma=somma+1
			j=j+1

	i=i+1

throughput_mean=np.asarray(throughput_mean)
res=fr_adi_res[(int(row['optimal_y'])-offset):(int(row['optimal_y'])+offset),(int(row['optimal_x'])-offset):(int(row['optimal_x'])+offset)]
r=row['fwhm_mean']/2.
out_file = open(os.path.join(path, 'contrast_curve.txt'),"w")
dist=[]
cont=[]

j=0
while r<offset:
	annulus_indices = get_annulus_segments(res, r, row['fwhm_mean']/2.)
	ann = res[annulus_indices]
	ann_abs=np.abs(ann)
	mean=np.mean(ann_abs[ann_abs != 0])
	stddev=np.std(ann_abs[ann_abs != 0])
	contrast=5*stddev/np.max(psf_norm)
	tcorr=tstudent.ppf(contrast,2*np.pi*r-1)
	tcorr=-tcorr
	contrastcorr=(tcorr*stddev*np.sqrt(1+1/(2*np.pi*r))+mean)/np.max(psf_norm)
	if (r<=throughput_mean[j,1]) or ((j+1)>=len(throughput_mean[:,0])):
		contrastcorr_trp=contrastcorr*1/throughput_mean[j,0]
		print(r,throughput_mean[j,0],throughput_mean[j,1])
	else :
		j=j+1
		contrastcorr_trp=contrastcorr*1/throughput_mean[j,0]
		print(r,throughput_mean[j,0],throughput_mean[j,1])
	out_file.write("%s,"%(str(r)))
	out_file.write("%s\n"%(str(contrastcorr_trp)))
	dist.append(r)
	cont.append(contrastcorr_trp)
	r=r+1

out_file.close()


fig, axes = plt.subplots(2,3)
axes = axes.flatten()
axes[0].imshow(cube[0], origin='lower')
axes[1].imshow(cube_emp[0], origin='lower')
axes[2].imshow(fr_adi_res, origin='lower')
axes[3].imshow(frame_square, origin='lower')
axes[4].imshow(psf_norm_crop, origin='lower')
plt.savefig('./figures/partial.png')

# cube_planet_free()