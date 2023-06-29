import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import cv2
import os

from src.record import _bytes_feature, _int64_feature
from src.preprocess import load_data


def translate_image(image, x_shift, y_shift):
	# Determine the dimensions of the image
	height, width = image.shape[:2]

	# Define the translation matrix
	translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

	# Apply the translation using warpAffine function
	translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

	return translated_image


def load_save_psfs(source, target_folder):
	folders = [os.path.join(source, x) for x in os.listdir(source)]

	if os.path.exists(os.path.join(target_folder, 'raw_data.pkl')):
		print('[INFO] Loading Normalized PSFs')
		with open(os.path.join(target_folder, 'raw_data.pkl'), 'rb') as file:
			output = pkl.load(file)
	else:
		print('[INFO] Normalizing PSFs')
		os.makedirs(target_folder, exist_ok=True)
		psfs, psfs_shape = [], []
		for root in folders:
			_, norm_psf, _ = load_data(root)
			psfs.append(norm_psf)
			psfs_shape.append(norm_psf.shape[-1])


		max_size = np.max(psfs_shape)

		all_pfs = []
		all_bands = []
		all_samples = []
		for k, psf in enumerate(psfs):
			for i, psf_lambda in enumerate(psf): 
				# reshaped_psf = cv2.resize(np.transpose(psf_lambda, [1, 2, 0]), (max_size, max_size))
				reshaped_psf = np.transpose(psf_lambda, [1, 2, 0])
				all_pfs.append(reshaped_psf)
				all_bands.append([i]*len(all_pfs))
				all_samples.append([k]*len(all_pfs))

		all_pfs = np.concatenate(all_pfs, axis=-1)
		all_pfs = np.transpose(all_pfs, [2,0,1])
		all_bands = np.concatenate(all_bands)
		all_samples = np.concatenate(all_samples)
		output = {'psf': all_pfs, 
				  'filters':all_bands, 
				  'ids':all_samples}

		with open(os.path.join(target_folder, 'raw_data.pkl'), 'wb') as file:
			pkl.dump(output, file)
	return output

def train_val_test_split(output, train_ptge, val_ptge, test_ptge):

	assert train_ptge + val_ptge + test_ptge == 1, 'train, val and test ptges must sum 1'

	n_psfs = len(output['psf'])

	n_train = int(n_psfs*train_ptge)
	n_val   = int(n_psfs*val_ptge)
	n_test  = int(n_psfs*(1.-(val_ptge+train_ptge))) 

	print(f'[INFO] Train: {n_train} Val:{n_val} Test:{n_test}')

	indices = np.arange(len(output['psf']))
	np.random.shuffle(indices)
	
	train = output['psf'][indices[:n_train]]
	val = output['psf'][indices[:n_val]]
	test = output['psf'][indices[:n_test]]

	return train, val, test


def create_records(opt):

	output = load_save_psfs(opt.source, opt.target)

	for fold in range(opt.folds):
		train, val, test = train_val_test_split(output, opt.train, opt.val, opt.test)	

		for subset_name, subset, nsamples in zip(['train', 'val', 'test'], 
												 [train, val, test], 
												 [opt.ntrain, opt.nval, opt.ntest]):

			target_subset = os.path.join(opt.target, f'fold_{fold}')
			os.makedirs(target_subset, exist_ok=True)
			with tf.io.TFRecordWriter(os.path.join(target_subset, subset_name+'.record')) as writer:
				for  _ in range(nsamples):
					x_shift = np.random.uniform(-subset.shape[-1]/3, subset.shape[-1]/3)
					y_shift = np.random.uniform(-subset.shape[-2]/3, subset.shape[-2]/3)

					num_psfs = subset.shape[0]
					random_index = np.random.randint(num_psfs)
					selected = subset[random_index]

					translated = translate_image(selected, x_shift, y_shift)
					translated = np.array(translated, dtype='float32')
					selected = np.array(selected, dtype='float32')
					x_bytes = translated.tobytes()
					y_bytes = selected.tobytes()

					feature = {
					'input': _bytes_feature(x_bytes),
					'output': _bytes_feature(y_bytes),
					'width': _int64_feature(translated.shape[-1]),
					'height': _int64_feature(translated.shape[-2]),
					}

					example = tf.train.Example(features=tf.train.Features(feature=feature))
					writer.write(example.SerializeToString())



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--target', default='./data/records/psf', type=str,
	                help='Target folder where records will be stored')
	parser.add_argument('--source', default='./data/real', type=str,
	                help='Source folder containing datasets with PSFs')
	parser.add_argument('--folds', default=1, type=int,
	                    help='Number of folds')
	parser.add_argument('--train', default=.5, type=float,
	                    help='Percentage (in decimals) of the total PSFs to be used as a part of the training set')
	parser.add_argument('--val', default=.25, type=float,
	                    help='Percentage (in decimals) of the total PSFs to be used as a part of the validation set')
	parser.add_argument('--test', default=.25, type=float,
	                    help='Percentage (in decimals) of the total PSFs to be used as a part of the testing set')
	parser.add_argument('--ntrain', default=20000, type=int,
	                    help='Number of samples to generate from the training PSFs')
	parser.add_argument('--nval', default=10000, type=int,
	                    help='Number of samples to generate from the validation PSFs')
	parser.add_argument('--ntest', default=100, type=int,
	                    help='Number of samples to generate from the testing PSFs')
	opt = parser.parse_args()
	create_records(opt)

# fig, axes = plt.subplots(1, 2, dpi=300)
# axes[0].imshow(selected)
# axes[1].imshow(translated)
# fig.savefig('./output/translated.png', bbox_inches='tight')
