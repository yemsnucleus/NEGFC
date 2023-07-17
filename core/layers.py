import tensorflow as tf
from scipy.ndimage import map_coordinates, shift

def shift_image(image, dydx_shift):
	image = tf.squeeze(image, -1)
	shifted = shift(image, shift=dydx_shift, order=1, mode='mirror')
	return shifted

class FluxPosRegressor(tf.keras.layers.Layer):

	def __init__(self, init_flux=None, **kwargs):
		super(FluxPosRegressor, self).__init__(**kwargs)
		self.init_flux = init_flux

	def build(self, input_shape):  # Create the state of the layer (weights)
		if self.init_flux is None:
			w_init = tf.random_normal_initializer()
			initial_value = w_init(shape=(input_shape[1], 1, 1, 1), dtype=tf.float32)
		else:
			initial_value= tf.ones([input_shape[1], 1, 1, 1], dtype=tf.float32) * self.init_flux

		self.flux = tf.Variable(
						initial_value=initial_value,
						trainable=True,
						name='flux_pred')

		noise_init = tf.zeros_initializer()
		self.noise = tf.Variable(
						initial_value=noise_init(shape=[input_shape[1], 1, 1, 1], dtype=tf.float32),
						trainable=True,
						name='noise')

		dydx_init = tf.random_normal_initializer()
		self.dydx = tf.Variable(
						initial_value=dydx_init([input_shape[1], 2], dtype=tf.float32),
						trainable=True,
						name='dydx')

	def call(self, inputs):  # Defines the computation from inputs to outputs
		inputs_shifted = tf.map_fn(lambda x: tf.numpy_function(shift_image, (x[0], x[1]), 
													   tf.float32), 
						   (inputs[0], self.dydx), 
						   fn_output_signature=tf.float32)		
		inputs_shifted = tf.reshape(inputs_shifted, tf.shape(inputs))
		scaled = inputs_shifted * self.flux + self.noise
		return scaled

	def get_config(self):
		config = super().get_config()
		config.update({
		"init_flux": self.init_flux})
		return config


class PosRegressor(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PosRegressor, self).__init__(**kwargs)

	def build(self, input_shape):  # Create the state of the layer (weights)
		print('input shape: ', input_shape)
		dxdy_init = tf.random_normal_initializer()
		self.dxdy = tf.Variable(
						initial_value=dxdy_init([input_shape[1], 2], dtype=tf.float32),
						trainable=True,
						name='theta')

	def call(self, inputs):  # Defines the computation from inputs to outputs
		inputs = tf.map_fn(lambda x: tf.numpy_function(shift_image, (x[0], x[1]), tf.float32), 
						   (inputs[0], self.dxdy), 
						   fn_output_signature=tf.float32)		
		inputs = tf.reshape(inputs, tf.shape(inputs))
		return inputs

	def get_config(self):
		config = super().get_config()
		return config

class FluxRegressor(tf.keras.layers.Layer):

	def __init__(self, init_flux=None, inp_shape=None, **kwargs):
		super(FluxRegressor, self).__init__(**kwargs)
		self.init_flux = init_flux
		self.inp_shape = [1]+list(inp_shape)
		print(self.inp_shape)
	def build(self, input_shape):  # Create the state of the layer (weights)
		if self.inp_shape is not None:
			input_shape = self.inp_shape

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
		"init_flux": self.init_flux,
		'inp_shape': self.inp_shape})
		return config