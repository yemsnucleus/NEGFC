import tensorflow as tf 

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LayerNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras        import Model
from .layer import TranslateCube

def build_input(window_size):
	inputs = {
		'windows': Input(shape=(window_size, window_size, 1),
						 batch_size=None,
						 name='window'),
		'psf': Input(shape=(window_size, window_size, 1),
					 batch_size=None,
					 name='psf'),
		'flux': Input(shape=(),
					 batch_size=None,
					 name='flux'),
	}
	return inputs



def create_embedding_model(window_size):

	input_placeholder = build_input(window_size)

	# Layers for cube
	conv_0 = Conv2D(64, (3, 3), activation='relu', input_shape=[window_size, window_size, 1])
	mp_0   = MaxPooling2D((2, 2))
	conv_1 = Conv2D(32, (3, 3), activation='relu')
	mp_1   = MaxPooling2D((2, 2))
	flat_layer = Flatten()

	# Layers for psf
	conv_2 = Conv2D(64, (3, 3), activation='relu', input_shape=[window_size, window_size, 1])
	mp_2   = MaxPooling2D((2, 2))
	conv_3 = Conv2D(32, (3, 3), activation='relu')
	mp_3   = MaxPooling2D((2, 2))
	flat_layer_2 = Flatten()

	# Layer to Merge
	layer_norm_0 = LayerNormalization()
	ffn_0 = Dense(128)
	ffn_1 = Dense(3)

	# Layer to separate parameters
	trans_layer = TranslateCube()

	# Network Architecture	
	x = conv_0(input_placeholder['windows'])
	x = mp_0(x)
	x = conv_1(x)
	x = mp_1(x)
	cube_emb = flat_layer(x)

	x = conv_2(input_placeholder['psf'])
	x = mp_2(x)
	x = conv_3(x)
	x = mp_3(x)
	psf_emb = flat_layer_2(x)

	x = tf.concat([cube_emb, psf_emb], 1)
	x = layer_norm_0(x)
	x = ffn_0(x)
	x = ffn_1(x)

	# Getting parameters
	dx = tf.slice(x, [0, 0], [-1, 1], name='dx')
	dy = tf.slice(x, [0, 1], [-1, 1], name='dy')
	dx = tf.math.maximum(dx, -window_size//3)
	dy = tf.math.maximum(dy, -window_size//3)
	dx = tf.math.minimum(dx, window_size//3)
	dy = tf.math.minimum(dy, window_size//3)
	dflux = tf.slice(x, [0, 2], [-1, 1], name='fluxes')

	# Moving PSF If necessary
	x = trans_layer((input_placeholder['psf'], dx, dy, window_size))

	flux = input_placeholder['flux'] + tf.squeeze(dflux, axis=-1)
	flux = tf.reshape(flux, [tf.shape(flux)[0], 1, 1])
	x = tf.multiply(x, flux)

	return CustomModel(inputs=input_placeholder, outputs=x, name="ConvNet")


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
            y_pred = self(x, training=True)
            loss = self.loss_fn(x['windows'], y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(x['windows'], y_pred)
        return {'loss': loss}
