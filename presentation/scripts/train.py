import tensorflow as tf
import argparse
import names
import os

from core.engine import first_guess, preprocess



def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	WEIGHTS_FOLDER = os.path.join(opt.p, opt.exp_name)
	
	table, cube, psf, rot_angles, backmoments = preprocess(opt.data, lambda_ch=0, load_preprocessed=True)
	

	# table = first_guess(table, cube, psf,
	# 					backmoments=backmoments, # 1st dim: flux / 2nd dim: std 
	#                     window_size=opt.ws, 
	#                     learning_rate=opt.lr, 
	#                     epochs=opt.epochs,
	#                     target_folder=WEIGHTS_FOLDER,
	#                     verbose=1)


	table = pd.read_csv(os.path.join(WEIGHTS_FOLDER, 'prediction.csv'))

	if opt.mcmc:
		print('[INFO] Running MCMC')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--p', default='./logs/f_dhtau', type=str,
					help='Project path to save logs and weigths')
	parser.add_argument('--exp-name', default='', type=str,
					help='experiment name')
	parser.add_argument('--data', default='./data/HCI', type=str,
					help='folder containing the cube, psfs, and rotational angles')
	parser.add_argument('--gpu', default='-1', type=str,
						help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--lr', default=1e-1, type=float,
						help='learning rate')
	parser.add_argument('--bs', default=16, type=int,
						help='Batch size')
	parser.add_argument('--patience', default=20, type=int,
						help='Earlystopping threshold in number of epochs')
	parser.add_argument('--epochs', default=1e6, type=int,
						help='Number of epochs')
	parser.add_argument('--ws', default=30, type=int,
						help='windows size of the PSFs')
	parser.add_argument('--mcmc', action='store_true', 
						help='if running Hamiltonian MCMC after getting first_guess outputs')

	opt = parser.parse_args()
	run(opt)