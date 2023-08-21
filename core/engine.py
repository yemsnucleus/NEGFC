import tensorflow as tf
import tensorflow_probability as tfp 
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os

from .model import create_model

from core.data import preprocess_and_save, get_companions, create_tf_dataset
from vip_hci.preproc.derotation import cube_derotate
from tensorflow.keras.optimizers import Adam, SGD
from multiprocessing import cpu_count
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.core.util import event_pb2


def rotate_cube(cube, rot_ang, derotate='tf', verbose=0):

	if derotate=='tf':
		if verbose: print('Using Tensorflow on GPU')

		rot_ang_deg = tf.experimental.numpy.deg2rad(rot_ang)
		if tf.rank(cube) == 3:
			cube = tf.expand_dims(cube, -1)

		res_derot = tfa.image.rotate(cube, -rot_ang_deg, 
									 interpolation='nearest', 
									 fill_mode='reflect')

	else:
		cores = multiprocessing.cpu_count()-2
		if verbose: print(f'Using VIP on {cores} cores')
		res_derot = cube_derotate(cube, 
								  rot_ang, 
								  nproc=cores, 
								  imlib='vip-fft', 
								  interpolation='nearneig')
	    
	return res_derot

@tf.function
def pca_tf(cube, out_size, rot_ang, ncomp=1, derotate='tf'):
	nframes = out_size[0]
	height = out_size[1]
	width = out_size[2]

	data = tf.reshape(cube, [nframes, height*width])
	data_mean = tf.reduce_mean(data, 1)
	data_centered = data - tf.expand_dims(data_mean, 1)
	data_centered = tf.reshape(data_centered, [nframes, height*width])
	s, u, v = tf.linalg.svd(data_centered)
	U =tf.transpose(u[:, :ncomp])
	transformed   = tf.matmul(U, data_centered)
	reconstructed = tf.transpose(tf.matmul(tf.transpose(transformed), U))
	residuals     = data_centered - reconstructed
	residuals_i   = tf.reshape(residuals, [nframes, height, width])
	reconstructed_i = tf.reshape(reconstructed, [nframes, height, width])

	residuals_i.set_shape([nframes, height, width])
	reconstructed_i.set_shape([nframes, height, width])

	res_derot = rotate_cube(residuals_i, rot_ang, derotate=derotate)
	res_derot = tf.reshape(res_derot, [tf.shape(cube)[0], tf.shape(cube)[1], tf.shape(cube)[2]])
	median = tfp.stats.percentile(res_derot, 50.0, 
	                              interpolation='midpoint', axis=0)
	median = tf.reshape(median, [tf.shape(cube)[1], tf.shape(cube)[2]])

	return median

def get_angle_radius(x, y, width, height):
	x = x - height//2
	y = y - width //2
	radius = np.sqrt(x**2+y**2) # radius
	angle  = np.arctan2(y, x)   # radians
	angle  = angle/np.pi*180    # degrees
	# Since atan2 return angles in between [0, 180] and [-180, 0],
	# we convert the angle to refers a system of 360 degrees
	theta0 = np.mod(angle, 360) 
	return radius, theta0

def preprocess(path, lambda_ch=0, p=25, load_preprocessed=True):
	if isinstance(path, str):
		cube, psf, rot_angles, table = preprocess_and_save(path, 
			lambda_ch=0, load_preprocessed=load_preprocessed)

	if isinstance(path, dict):
		cube = path['cube']
		psf = path['psf']
		rot_angles = path['rot_angles']
		table = None

	cube = cube_derotate(cube, rot_angles, nproc=4, imlib='opencv', interpolation='nearneig')
	
	# ---- estimating noise ----
	max_val = np.percentile(cube, p)
	mask_in = np.array(cube<max_val, dtype='float')
	cube_filtered = cube * mask_in
	
	mean_per_frame = np.mean(cube_filtered, axis=(1,2))
	std_per_frame  = np.std(cube_filtered, axis=(1,2))
	moments = np.vstack([mean_per_frame, std_per_frame]).T
	return table, cube, psf, rot_angles, moments

def inference_step(cube, psf, x, y, model_path, window_size):
	companion = get_companions(cube, x=x, y=y, window_size=window_size)
	psf       = get_companions(psf, x=psf.shape[-1]//2, y=psf.shape[-1]//2, window_size=window_size)
	loader, input_shape = create_tf_dataset(psf, companion, batch_size=1, repeat=1)

	model = create_model(input_shape=input_shape)
	model.load_weights(os.path.join(model_path, 'weights')).expect_partial()
	y_pred = model.predict(loader)
	fluxes = model.trainable_variables[0].numpy()
	
	return y_pred[0, ...,0], companion, np.squeeze(fluxes), model

def get_callbacks(log_dir, loss_precision):
	es  = tf.keras.callbacks.EarlyStopping(monitor='loss', 
										   patience=10, 
										   min_delta=loss_precision,
										   restore_best_weights=True)
	if log_dir == 'temp':
		return es

	tb  = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	return [es, tb]

def first_guess(table, cube, psf, backmoments, window_size=30, learning_rate=1e-2, 
				epochs=1e6, n_jobs=None, 
				target_folder=None, verbose=0, 
				loss_precision=0.):
	if n_jobs is None:
		n_jobs = int(cpu_count()//2)

	if target_folder is not None:
		os.makedirs(target_folder, exist_ok=True)
	else:
		target_folder = 'temp'
	
	optimal_fluxes, optimal_xs, optimal_ys = [], [], []
    
	for index, row in table.iterrows():
		print('[INFO] Training (x, y) = ({:.2f} {:.2f})'.format(row['x'], row['y']))
		companion = get_companions(cube, x=row['x'], y=row['y'], window_size=window_size)
		psf       = get_companions(psf, x=psf.shape[-1]//2, y=psf.shape[-1]//2, window_size=window_size)

		loader, input_shape = create_tf_dataset(psf, companion, batch_size=1, repeat=1)

		model = create_model(input_shape=input_shape, init_flux=row['flux'])

		model.compile(backmoments=backmoments, 
					  fwhm=row['fwhm_mean'], 
					  optimizer=Adam(learning_rate))

		ckbs = get_callbacks(os.path.join(target_folder, f'model_{index}', 'logs'),
							 loss_precision=loss_precision)

		hist = model.fit(loader, epochs=int(epochs), 
						 callbacks=ckbs, 
						 workers=n_jobs, 
						 use_multiprocessing=True, 
						 verbose=verbose)

		if target_folder != 'temp':
			model.save_weights(os.path.join(target_folder, f'model_{index}', 'weights'))

		best_epoch = np.argmin(hist.history['loss'])
		best_flux = hist.history['flux'][best_epoch]
		
		dydx = model.trainable_variables[-1]
		dydx = tf.reduce_mean(dydx, 0)
		
		opt_x = row['x'] + dydx[1]
		opt_y = row['y'] + dydx[0]
		optimal_xs.append(opt_x.numpy()) 
		optimal_ys.append(opt_y.numpy())
		optimal_fluxes.append(best_flux)

	table['optimal_flux'] = optimal_fluxes
	table['optimal_x'] = optimal_xs
	table['optimal_y'] = optimal_ys
	table = table.reset_index()
	table['index'] = table['index'].astype(int)
	if target_folder != 'temp':
		table.to_csv(os.path.join(target_folder, 'prediction.csv'), index=False)
		
	return table


def get_metrics(path_logs, metric_name='epoch_loss', full_logs=True, show_keys=False):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][-1]
    path_train = os.path.join(path_logs, train_logs)

    if full_logs:
        ea = event_accumulator.EventAccumulator(path_train, size_guidance={'tensors': 0})
    else:
        ea = event_accumulator.EventAccumulator(path_train)

    ea.Reload()

    if show_keys:
        print(ea.Tags())

    metrics = pd.DataFrame([(w,s,tf.make_ndarray(t))for w,s,t in ea.Tensors(metric_name)],
                columns=['wall_time', 'step', 'value'])
    return metrics