import tensorflow as tf
from .layers import FluxPosRegressor, PosRegressor,FluxRegressor
from .losses import reduce_moments


def create_model(input_shape, init_flux=None):
	psf = tf.keras.Input(shape=input_shape, 
						 dtype=tf.float32)
	flux_regressor = FluxPosRegressor(init_flux=init_flux)

	fake_comp = flux_regressor(psf)
	return CustomModel(model_type='flux_pos', inputs=psf, outputs=fake_comp, name='Regressor')

class CustomModel(tf.keras.Model):
	def __init__(self, model_type='flux_pos', **kwargs):
		super(CustomModel, self).__init__(**kwargs)
		self.model_type = model_type

	def compile(self, backmoments=None, **kwargs):
		super(CustomModel, self).compile(**kwargs)
		self.backmoments = backmoments

	def train_step(self, data):
		# Unpack the data. Its structure depends on your model and
		# on what you pass to `fit()`.
		x, y = data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)  # Forward pass
			mean_loss, std_loss = reduce_moments(y_true=y, 
												 y_pred=y_pred, 
												 moments=self.backmoments)
			loss = mean_loss + std_loss

		# Compute gradients
		trainable_vars = self.trainable_variables
		
		gradients = tape.gradient(loss, trainable_vars)
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		flux  = tf.reduce_mean(trainable_vars[0])
		noise = tf.reduce_mean(trainable_vars[1])

		metrics = {
				'loss': loss,
				'res_mean': mean_loss,
				'res_std': std_loss,
				'flux': flux,
				'noise': noise
			}


		return metrics