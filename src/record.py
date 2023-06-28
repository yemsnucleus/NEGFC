import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import numpy as np
import os

from functools import partial
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def cut_patch(inputs, x, y, window_size):
	frame, angle = inputs

	width, height = frame.shape
	pil_img = Image.fromarray(frame)
	rot_img = pil_img.rotate(angle)
	rot_img = np.array(rot_img)

	if window_size % 2 != 0:
		window_size = window_size+1

	start_w = int(np.maximum(0, y - window_size//2))
	start_h = int(np.maximum(0, x - window_size//2))

	end_w   = int(np.minimum(width, y + window_size//2))
	end_h   = int(np.minimum(height, x + window_size//2))

	patch = rot_img[start_w:end_w, start_h:end_h]
	return patch

def save_subset_records(cube, psf, rot_angles, table, output_path, window_size, njobs):
	for index, row in table.iterrows():
		pool = multiprocessing.Pool(processes=njobs)
		partial_cut_patch = partial(cut_patch, x=row['x'], y=row['y'], window_size=window_size)
		cube_crop = pool.map(partial_cut_patch, zip(cube, -rot_angles))
		pool.close()
		pool.join()
		candidate = np.array(cube_crop, dtype='float32')

		pool = multiprocessing.Pool(processes=njobs)
		partial_cut_patch = partial(cut_patch, x=psf.shape[-1]//2, y=psf.shape[-1]//2, window_size=window_size)
		psf_crop = pool.map(partial_cut_patch, zip(psf, -rot_angles))
		pool.close()
		pool.join()
		psf_crop = np.array(psf_crop, dtype='float32')
		
		with tf.io.TFRecordWriter(os.path.join(output_path, '{}.record'.format(index))) as writer:

			psf_bytes  = psf_crop.tobytes()
			cube_bytes = candidate.tobytes()
			
			depth_f = candidate.shape[-0] # number of frames
			depth_p = psf_crop.shape[0] # number of psfs

			width_f = candidate.shape[-1] # frame size
			width_p = psf_crop.shape[-1] # psf size

			feature = {
			'cube': _bytes_feature(cube_bytes),
			'psf': _bytes_feature(psf_bytes),
			'index': _int64_feature(index),
			'depth_f': _int64_feature(depth_f),
			'depth_p': _int64_feature(depth_p),
			'width_f': _int64_feature(width_f),
			'width_p': _int64_feature(width_p),
			}

			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())

def save_records(cube, psf, rot_angles, table, output_path='./data/records', snr_threshold=10, 
				 window_size=None, njobs=None, n_folds=1, train_val_test=(0.6, 0.2, 0.2)):

	if window_size is None:
		window_size = psf.shape[-1]
	assert psf.shape[-1] >= window_size, 'window size must be lower than PSF size ({}) '.format(psf.shape)

	ncores = multiprocessing.cpu_count()
	if njobs is None:
		njobs = ncores//2
	assert njobs<=ncores, 'inssuficient resources. njobs must be lower than {} cores'.format(ncores)
	assert np.sum(train_val_test) == 1., 'train_val_test must sum 1'


	# ==== filter by signal to noise ratio
	table = table[table['snr'] > snr_threshold]

	# ==== folds and train-val-test
	for fold in range(n_folds):
		table = table.sample(frac=1)
		train_samples = table.sample(frac=train_val_test[0])
		rest = table[~table.index.isin(train_samples.index)]
		valfrac = (train_val_test[1])/(1. - train_val_test[0])
		
		val_samples  = rest.sample(frac=valfrac)
		test_samples = rest[~rest.index.isin(val_samples.index)]

		for name, frame in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
			new_output_path = os.path.join(output_path, f'fold_{fold}', name)
			os.makedirs(new_output_path, exist_ok=True)
			save_subset_records(cube, psf, rot_angles, frame, new_output_path, window_size, njobs)


def parse_candidate(serialized_example):
	feature_description = {
		'cube': tf.io.FixedLenFeature([], tf.string),
		'psf': tf.io.FixedLenFeature([], tf.string),
		'index': tf.io.FixedLenFeature([], tf.int64),
		'depth_f': tf.io.FixedLenFeature([], tf.int64),
		'depth_p': tf.io.FixedLenFeature([], tf.int64),
		'width_f': tf.io.FixedLenFeature([], tf.int64),
		'width_p': tf.io.FixedLenFeature([], tf.int64),
	}
	example = tf.io.parse_single_example(serialized_example, feature_description)

	# Decode the cube from bytes
	cube = tf.io.decode_raw(example['cube'], tf.float32)
	psf = tf.io.decode_raw(example['psf'], tf.float32)

	cube = tf.reshape(cube, [example['depth_f'], example['width_f'], example['width_f'], 1])
	psf = tf.reshape(psf, [example['depth_p'], example['width_p'], example['width_p'], 1])
	psf = tf.slice(psf, [0, 0, 0, 0], [1, -1, -1, -1])
	psf = tf.tile(psf, [example['depth_f'], 1, 1, 1])
	return {'cube':cube, 'psf':psf}

def augment(inputs):
	rnd_prob_rot = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
	if rnd_prob_rot> 0.5:
		rnd_k = tf.random.uniform([], minval=1, maxval=5, dtype=tf.int32)
		inputs['cube'] = tf.map_fn(lambda x: tf.image.rot90(x, k=rnd_k), inputs['cube'])
		inputs['psf'] = tf.map_fn(lambda x: tf.image.rot90(x, k=rnd_k), inputs['psf'])

	rnd_prob_left_right = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
	if rnd_prob_left_right > 0.5:
		inputs['cube'] = tf.map_fn(lambda x: tf.image.flip_left_right(x), inputs['cube'])
		inputs['psf'] = tf.map_fn(lambda x: tf.image.flip_left_right(x), inputs['psf'])
	
	rnd_prob_up_down = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
	if rnd_prob_up_down > 0.5:
		inputs['cube'] = tf.map_fn(lambda x: tf.image.flip_up_down(x), inputs['cube'])
		inputs['psf'] = tf.map_fn(lambda x: tf.image.flip_up_down(x), inputs['psf'])

	return inputs

def load_records(folder, batch_size=2, repeat=1):
	record_files = [ os.path.join(folder, x) for x in os.listdir(folder) if x.endswith('.record') ]	

	dataset = tf.data.TFRecordDataset(record_files)
	dataset = dataset.map(parse_candidate)
	dataset = dataset.repeat(repeat)
	dataset = dataset.map(augment)
	dataset = dataset.batch(batch_size)
	dataset = dataset.cache()
	return dataset


