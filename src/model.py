import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras        import Model
from .layer import TranslateCube, PositionRegressor, FluxRegressor, CubeConvBlock, PSFConvBlock, FluxPosRegressor, ConvolutionalLayer
from .format_data import DTYPE

def build_input(window_size):
	inputs = {
		'windows': Input(shape=(window_size, window_size, 1),
						 batch_size=None, dtype=DTYPE,
						 name='window'),
		'psf': Input(shape=(window_size, window_size, 1),
					 batch_size=None, dtype=DTYPE,
					 name='psf'),
		'flux': Input(shape=(),
					 batch_size=None, dtype=DTYPE,
					 name='flux'),
	}
	return inputs


def create_convnet(window_size):

    input_placeholder = build_input(window_size)
    
    # Layers
    cnn_layer = ConvolutionalLayer(window_size, name='cubeCNN')
    shift_op = TranslateCube(name='shift')
    
    # Network
    dx, dy, flux = cnn_layer(input_placeholder)
    
    flux = tf.cast(flux, DTYPE)
    flux = tf.reshape(flux, [-1, 1, 1, 1])
    flux = tf.tile(flux, [1, window_size, window_size, 1]) 
    
    x = input_placeholder['psf'] * flux
    x = shift_op((x, dx, dy, window_size))
    x = tf.cast(x, DTYPE)
    x = tf.expand_dims(x, axis=-1)
       
    x_params = (dx, dy, flux)
    return CustomModel(inputs=input_placeholder, outputs=(x, x_params), name="ConvNet2")


def create_embedding_model(window_size):

    input_placeholder = build_input(window_size)
    
    # Layers
    cube_cnn = CubeConvBlock(window_size, name='cubeCNN')
    psf_cnn  = PSFConvBlock(window_size, name='psfCNN')
    dxdy_reg = PositionRegressor(units=128, name='dxdyREG')
    flux_reg = FluxRegressor(units=128, name='fluxREG')
    shift_op = TranslateCube(name='shift')

    # Network Architecture
    cube_emb = cube_cnn(input_placeholder['windows'])
    psf_emb = psf_cnn(input_placeholder['psf'])
    dx, dy = dxdy_reg(cube_emb)
    dflux  = flux_reg(psf_emb)

    x = shift_op((input_placeholder['psf'], dx, dy, window_size))
    x = tf.cast(x, DTYPE)
    x = tf.expand_dims(x, axis=-1)
    
    flux = tf.cast(dflux, DTYPE) #input_placeholder['flux'] + tf.cast(dflux, DTYPE)
    flux = tf.reshape(flux, [-1, 1, 1, 1])
    flux = tf.tile(flux, [1, window_size, window_size, 1])

    x = x * flux    
    x_params = (dx, dy, flux)
    return CustomModel(inputs=input_placeholder, outputs=(x, x_params), name="ConvNet")

def create_flux_model(window_size):

	input_placeholder = build_input(window_size)

	# Layers
	cube_cnn = CubeConvBlock(window_size, name='cubeCNN')
	psf_cnn  = PSFConvBlock(window_size, name='psfCNN')
	regresor = FluxPosRegressor(128)

	# Network Architecture
	cube_emb = cube_cnn(input_placeholder['windows'])
	psf_emb = psf_cnn(input_placeholder['psf'])
	flux  = regresor([cube_emb, psf_emb])
	flux = tf.reshape(flux, [-1, 1, 1, 1])
	flux = tf.tile(flux, [1, window_size, window_size, 1])
	x = input_placeholder['psf'] * flux

	return CustomModel(inputs=input_placeholder, outputs=(x, flux), name="ConvNet")


class CustomModel(tf.keras.Model):
    '''
    Custom functional model
    '''
    def compile(self, loss_fn, **kwargs):
        super(CustomModel, self).compile(**kwargs)
        self.loss_fn = loss_fn
        
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, params = self(x, training=True)
            loss = self.loss_fn(x, y, params, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.loss_fn(x, y, params, y_pred)
        return {'loss': loss}

    @tf.function
    def predict_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, params  = self(x, training=True)
            loss = self.loss_fn(x, y, params, y_pred)
                      
#         unique_ids, idx = tf.unique(y['ids'])  # Get unique values and indices
#         output_pred = tf.dynamic_partition(y_pred, idx, tf.shape(unique_ids)[0])
        fluxes = tf.reduce_max(params[-1], axis=[1,2,3])
#         output_flux = tf.dynamic_partition(fluxes, idx, tf.shape(unique_ids)[0])
        return y_pred, (params[0], params[1], fluxes)