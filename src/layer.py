import tensorflow_addons as tfa
import tensorflow as tf

from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LayerNormalization

@tf.function
def translate_image(inputs):
	inp_tensor, dx, dy, winsize = inputs
	cords = tf.stack([dx, dy], 1)
	inp_tensor = tf.squeeze(inp_tensor, -1)
	cords = tf.squeeze(cords)
	out = tfa.image.translate(inp_tensor, cords)
	return out

class TranslateCube(Layer):
	def __init__(self, name='shift'):
		super(TranslateCube, self).__init__(name=name)

	def call(self, inputs):
		images = inputs[0]
		dx = inputs[1]
		dy = inputs[2]
		winsize = inputs[3]
		winsize = tf.tile(tf.expand_dims(winsize, 0), [tf.shape(dx)[0]])

		x = tf.map_fn(lambda x: translate_image(x), (images, dx, dy, winsize), fn_output_signature=tf.float32)
		return x

class PositionRegressor(Layer):
    def __init__(self, units, name='coords_reg'):
        super(PositionRegressor, self).__init__(name=name)
        self.units = units
        self.norm = LayerNormalization()
        self.ffn_0 = Dense(units)
        self.ffn_1 = Dense(2, name='regressor')

    def call(self, inputs):
        x = self.norm(inputs) 
        x = self.ffn_0(x) 
        x = self.ffn_1(x)
        dx = tf.slice(x, [0, 0], [-1, 1], name='dx')
        dy = tf.slice(x, [0, 1], [-1, 1], name='dy')
        return dx, dy
    
class FluxRegressor(Layer):
    def __init__(self, units, name='flux_reg'):
        super(FluxRegressor, self).__init__(name=name)
        self.units = units
        self.norm = LayerNormalization()
        self.ffn_0 = Dense(units)
        self.ffn_1 = Dense(1, name='regressor')

    def call(self, inputs):
        x = self.norm(inputs) 
        x = self.ffn_0(x) 
        x = self.ffn_1(x)
        x = tf.squeeze(x, axis=-1)
        return x
    
class CubeConvBlock(Layer):
    def __init__(self, window_size, name='cube_cnn'):
        super(CubeConvBlock, self).__init__(name=name)
        self.window_size = window_size
        self.conv_0 = Conv2D(64, (3, 3), activation='relu', input_shape=[window_size, window_size, 1])
        self.mp_0   = MaxPooling2D((2, 2))
        self.conv_1 = Conv2D(32, (3, 3), activation='relu')
        self.mp_1   = MaxPooling2D((2, 2))
        self.flat_layer = Flatten()

    def call(self, inputs):
        x = self.conv_0(inputs)
        x = self.mp_0(x)
        x = self.conv_1(x)
        x = self.mp_1(x)
        x = self.flat_layer(x)
        return x
    
class PSFConvBlock(Layer):
    def __init__(self, window_size, name='psf_cnn'):
        super(PSFConvBlock, self).__init__(name=name)
        self.window_size = window_size
        self.conv_0 = Conv2D(64, (3, 3), activation='relu', input_shape=[window_size, window_size, 1])
        self.mp_0   = MaxPooling2D((2, 2))
        self.conv_1 = Conv2D(32, (3, 3), activation='relu')
        self.mp_1   = MaxPooling2D((2, 2))
        self.flat_layer = Flatten()

    def call(self, inputs):
        x = self.conv_0(inputs)
        x = self.mp_0(x)
        x = self.conv_1(x)
        x = self.mp_1(x)
        x = self.flat_layer(x)
        return x