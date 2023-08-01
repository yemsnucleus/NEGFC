import tensorflow as tf
import multiprocessing
import vip_hci as vip
import numpy as np
import argparse
import names
import time
import os

from core.engine import first_guess, preprocess



def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


	if opt.back == 'tf':

		for ws in [15, 25, 50]:
			TARGET_FOLDER = os.path.join(opt.p, 'tf', opt.exp_name, 'w{}'.format(ws))
			
			start = time.time()
			table, cube, psf, rot_angles, backmoments = preprocess(opt.data, 
																   lambda_ch=0,
																   load_preprocessed=False)
			
			table = table[table['snr']>4]
			
			table = first_guess(table, cube, psf,
								backmoments=backmoments, # 1st dim: flux / 2nd dim: std 
								window_size=ws, 
								learning_rate=1e0, 
								epochs=1e6,
								target_folder=TARGET_FOLDER,
								verbose=1)
			
			end = time.time()
			table['elapsed'] = [end-start]
			table.to_csv(os.path.join(TARGET_FOLDER, 'prediction.csv'), index=False)

	if opt.back == 'vip':
		for fmerit in ['stddev', 'sum']:
			TARGET_FOLDER = os.path.join(opt.p, 'vip', opt.exp_name, fmerit)
			os.makedirs(TARGET_FOLDER, exist_ok=True)

			start = time.time()
			table, cube, psf, rot_angles, backmoments = preprocess(opt.data, 
																   lambda_ch=0,
																   load_preprocessed=False)
			table = table[table['snr']>4]
			results = vip.fm.negfc_simplex.firstguess(cube,
													  angs=rot_angles,
													  psfn=psf[0],
													  fmerit = fmerit,
													  ncomp=1,
													  planets_xy_coord=[(table.iloc[0].x, table.iloc[0].y)],
													  imlib='opencv',
													  fwhm=table.iloc[0].fwhm_mean,
													  simplex=True,
													  verbose=False,
													  annulus_width=5,
													  aperture_radius=5,
													  f_range=np.arange(table.iloc[0].flux -10, table.iloc[0].flux*2, 20),
													 algo_options={
														 'nproc': multiprocessing.cpu_count()//2,
														 'imlib': 'opencv'
													 },
													  mu_sigma=False)

			end = time.time()
		
			centerx = cube.shape[-1]//2
			centery = cube.shape[-2]//2
			r = results[0][0]
			theta = results[1][0]
			flux = results[2][0]

			plcnx = r * np.sin(np.deg2rad(theta)) + centerx
			plcny = r * np.cos(np.deg2rad(theta)) + centery

			table['elapsed'] = [end-start]
			table['optimal_x'] = [plcnx]
			table['optimal_y'] = [plcny]
			table['optimal_flux'] = [flux]

			table.to_csv(os.path.join(TARGET_FOLDER, 'prediction.csv'), index=False)	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--p', default='./logs/comparison', type=str,
					help='Project path to save logs and weigths')
	parser.add_argument('--back', default='tf', type=str,
					help='what framework use to perform first_guess (tf/vip)')
	parser.add_argument('--exp-name', default='foo', type=str,
					help='experiment name')

	parser.add_argument('--data', default='./data/HCI', type=str,
					help='folder containing the cube, psfs, and rotational angles')

	parser.add_argument('--gpu', default='-1', type=str,
						help='GPU to be used. -1 means no GPU will be used')

	opt = parser.parse_args()
	run(opt)