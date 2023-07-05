import tensorflow as tf
import pandas as pd
import numpy as np
import os

from .model import create_model
from core.data import preprocess_and_save, get_companions, create_tf_dataset
from vip_hci.preproc.derotation import cube_derotate
from tensorflow.keras.optimizers import Adam
from multiprocessing import cpu_count
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.core.util import event_pb2

def preprocess(path, lambda_ch=0):
	cube, psf, rot_angles, table = preprocess_and_save(path, lambda_ch=0)
	cube = cube_derotate(cube, rot_angles, nproc=4, imlib='opencv', interpolation='nearneig')
	return table, cube, psf


def inference_step(cube, psf, x, y, model_path, window_size):
	companion = get_companions(cube, x=x, y=y, window_size=window_size)
	psf       = get_companions(psf, x=psf.shape[-1]//2, y=psf.shape[-1]//2, window_size=window_size)
	loader, input_shape = create_tf_dataset(psf, companion, batch_size=1, repeat=1)

	model = create_model(input_shape=input_shape)
	model.load_weights(os.path.join(model_path, 'weights')).expect_partial()
	y_pred = model.predict(loader)
	fluxes = model.trainable_variables[0].numpy()
	
	return y_pred[0,...,0], companion, np.squeeze(fluxes)

def first_guess(table, cube, psf, window_size=30, learning_rate=1e-2, epochs=1e6, n_jobs=None, target_folder='.', verbose=0):
	if n_jobs is None:
		n_jobs = int(cpu_count()//2)

	os.makedirs(target_folder, exist_ok=True)

	optimal_fluxes = []
	for index, row in table.iterrows():
		print('[INFO] Training (x, y) = ({:.2f} {:.2f})'.format(row['x'], row['y']))
		companion = get_companions(cube, x=row['x'], y=row['y'], window_size=window_size)
		psf       = get_companions(psf, x=psf.shape[-1]//2, y=psf.shape[-1]//2, window_size=window_size)

		loader, input_shape = create_tf_dataset(psf, companion, batch_size=1, repeat=1)

		model = create_model(input_shape=input_shape, init_flux=row['flux'])

		model.compile(optimizer=Adam(learning_rate))

		es  = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
		tb  = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(target_folder, f'model_{index}', 'logs'))

		hist = model.fit(loader, epochs=int(epochs), callbacks=[es, tb], workers=n_jobs, 
						 use_multiprocessing=True, verbose=verbose)

		model.save_weights(os.path.join(target_folder, f'model_{index}', 'weights'))

		best_epoch = np.argmin(hist.history['loss'])
		best_flux = hist.history['flux'][best_epoch]
		optimal_fluxes.append(best_flux)

	table['optimal_flux'] = optimal_fluxes
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