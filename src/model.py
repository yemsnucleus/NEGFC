import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras        import Model
from .layer import TranslateCube, PositionRegressor, FluxRegressor, CubeConvBlock, PSFConvBlock, FluxPosRegressor
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
    
    flux = input_placeholder['flux'] + tf.cast(dflux, DTYPE)
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
            y_pred, _ = self(x, training=True)
            loss = self.loss_fn(x['windows'], y_pred, fwhm=y['fwhm'])
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.loss_fn(x['windows'], y_pred, fwhm=y['fwhm'])
        return {'loss': loss}

    @tf.function
    def predict_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, params  = self(x, training=True)
            loss = self.loss_fn(x['windows'], y_pred, fwhm=y['fwhm'])
        return y_pred, params