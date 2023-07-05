import tensorflow as tf


class FluxRegressor(tf.keras.layers.Layer):

	def __init__(self, init_flux=None):
		super(FluxRegressor, self).__init__()
		self.init_flux = init_flux

	def build(self, input_shape):  # Create the state of the layer (weights)
		print('input shape: ', input_shape)
		if self.init_flux is None:
			w_init = tf.random_normal_initializer()
			initial_value = w_init(shape=(input_shape[1], 1, 1, 1), dtype=tf.float32)
		else:
			initial_value= tf.ones([input_shape[1], 1, 1, 1], dtype=tf.float32) * self.init_flux

		self.flux = tf.Variable(
						initial_value=initial_value,
						trainable=True,
						name='flux_pred')

		b_init = tf.zeros_initializer()
		self.noise = tf.Variable(
						initial_value=b_init(shape=(input_shape[1], 1, 1, 1), dtype=tf.float32),
						trainable=True,
						name='noise')

	def call(self, inputs):  # Defines the computation from inputs to outputs
		scaled = inputs * self.flux + self.noise
		return scaled

	def get_config(self):
		config = super().get_config()
		config.update({
		"init_flux": self.init_flux})
		return config

def create_model(input_shape, init_flux=None):
	psf = tf.keras.Input(shape=input_shape, 
						 dtype=tf.float32)
	flux_regressor = FluxRegressor(init_flux=init_flux)
	fake_comp = flux_regressor(psf)
	return CustomModel(inputs=psf, outputs=fake_comp, name='Regressor')

def reduce_std(y_true, y_pred):
	residuals = tf.pow(y_true - y_pred, 2)
	std = tf.math.reduce_std(residuals, axis=[2, 3, 4])
	return tf.reduce_mean(std)


class CustomModel(tf.keras.Model):
	def compile(self, **kwargs):
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

		# Return a dict mapping metric names to current value
		pred_flux = tf.reduce_mean(trainable_vars[0])
		pred_std = tf.reduce_mean(trainable_vars[1])

		return {'loss': loss, 'flux':pred_flux, 'std': pred_std}
