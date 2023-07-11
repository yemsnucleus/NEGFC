import tensorflow as tf
from .layers import FluxPosRegressor, PosRegressor,FluxRegressor

# ====== MODELS ======
def create_model(input_shape, init_flux=None):
	psf = tf.keras.Input(shape=input_shape, 
						 dtype=tf.float32)
	flux_regressor = FluxPosRegressor(init_flux=init_flux)

	fake_comp = flux_regressor(psf)
	return CustomModel(model_type='flux_pos', inputs=psf, outputs=fake_comp, name='Regressor')

def create_pos_model(input_shape):
	psf = tf.keras.Input(shape=input_shape, 
						 dtype=tf.float32)
	
	pos_regressor = PosRegressor(name='pos_reg')

	fake_comp = pos_regressor(psf)
	return CustomModel(model_type='pos', inputs=psf, outputs=fake_comp, name='PosRegressor')

def create_flux_model(input_shape, init_flux=None, pos_model=None):
	psf_plhd = tf.keras.Input(shape=input_shape, 
						 dtype=tf.float32)
	
	flux_regressor = FluxRegressor(init_flux=init_flux, inp_shape=input_shape)

	if pos_model is not None:
		pos_regressor = pos_model.get_layer('pos_reg')
		pos_regressor.trainable = False
		x = pos_regressor(psf_plhd, training=False)
	else:
		x = psf_plhd

	fake_comp = flux_regressor(x)

	return CustomModel(model_type='flux', inputs=psf_plhd, outputs=fake_comp, name='FluxRegressor')


# ====== TRAINING FUNCTIONS ======
def format_flux_pos_model_output(loss, trainable_vars):
	# Return a dict mapping metric names to current value
	pred_flux = tf.reduce_mean(trainable_vars[0])
	pred_std = tf.reduce_mean(trainable_vars[1])
	return {'loss': loss, 'flux':pred_flux, 'std': pred_std}

def format_pos_model_output(loss, trainable_vars):
	# Return a dict mapping metric names to current value
	pred = tf.reduce_mean(trainable_vars, axis=[0, 1])
	return {'loss': loss, 'dx': pred[0], 'dy':pred[1]}

def reduce_std(y_true, y_pred):
	residuals = tf.pow(y_true - y_pred, 2)
	std = tf.math.reduce_std(residuals, axis=[2, 3, 4])
	return tf.reduce_mean(std)

class CustomModel(tf.keras.Model):
	def __init__(self, model_type='flux_pos', **kwargs):
		super(CustomModel, self).__init__(**kwargs)
		self.model_type = model_type

	def compile(self,**kwargs):
		super(CustomModel, self).compile(**kwargs)

	def train_step(self, data):
		# Unpack the data. Its structure depends on your model and
		# on what you pass to `fit()`.
		x, y = data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
			loss = reduce_std(y_true=y, y_pred=y_pred)

		# Compute gradients
		trainable_vars = self.trainable_variables
		
		gradients = tape.gradient(loss, trainable_vars)
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		if self.model_type == 'flux_pos' or self.model_type == 'flux':
			return format_flux_pos_model_output(loss, trainable_vars)	
		if self.model_type == 'pos':
			return format_pos_model_output(loss, trainable_vars)