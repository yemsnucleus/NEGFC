import tensorflow_addons as tfa
import tensorflow as tf

from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LayerNormalization

@tf.function
def translate(image, dx, dy, winsize):
    dx = tf.cast(dx, tf.int32)
    dy = tf.cast(dy, tf.int32)
        
    pad_x = tf.zeros([image.shape[0], tf.abs(dx), 1])
    pad_y = tf.zeros([1, image.shape[1]+tf.abs(dx), tf.abs(dy)])
    pad_y = tf.transpose(pad_y)
            
    start_x=0 
    start_y=0
    if dx > 0:
        image = tf.concat([pad_x, image], 1)
    if dx < 0:
        image= tf.concat([image, pad_x], 1)
        start_x=start_x+tf.abs(dx)

    if dy < 0:
        image = tf.concat([pad_y, image], 0)
    if dy > 0:
        image = tf.concat([image, pad_y], 0)
        start_y = start_y+tf.abs(dy)
    
    transformed = tf.slice(image, [start_y, start_x, 0], [winsize, winsize, -1])
    return transformed

class TranslationLayer(Layer):
    def __init__(self, winsize, name='translation'):
        super(TranslationLayer, self).__init__(name=name)
        self.winsize = winsize
        
    def call(self, inputs, dx, dy):
        y_pred = tf.map_fn(lambda x: translate(x[0], x[1], x[2], self.winsize), 
                          (inputs, dx, dy), 
                          fn_output_signature=tf.float32)
        return y_pred
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "winsize": self.winsize,
        })
        return config
    
class PositionRegressor(Layer):
    def __init__(self, name='coords_reg'):
        super(PositionRegressor, self).__init__(name=name)
        self.norm = LayerNormalization()
        self.ffn_0 = Dense(128)
        self.ffn_1 = Dense(64)
        self.ffn_2 = Dense(2, name='regressor')

    def call(self, inputs):
        x = self.norm(inputs) 
        x = self.ffn_0(x) 
        x = self.ffn_1(x)
        x = self.ffn_2(x)
        # dx = tf.slice(x, [0, 0], [-1, 1], name='dx')
        # dy = tf.slice(x, [0, 1], [-1, 1], name='dy')     
	
        # dx = tf.clip_by_value(dx, clip_value_min=20., clip_value_max=20.)
        # dy = tf.clip_by_value(dy, clip_value_min=20., clip_value_max=20.)
        
        # dx = tf.squeeze(dx, axis=-1)
        # dy = tf.squeeze(dy, axis=-1)
        return x
    
class ConvBlock(Layer):
    def __init__(self, window_size, name='cube_cnn'):
        super(ConvBlock, self).__init__(name=name)
        self.window_size = window_size
        self.conv_0 = Conv2D(128, (2, 2), activation='relu', input_shape=[window_size, window_size, 1])
        self.mp_0   = MaxPooling2D((2, 2))
        self.conv_1 = Conv2D(64, (2, 2), activation='relu')
        self.mp_1   = MaxPooling2D((2, 2))
        self.conv_2 = Conv2D(32, (2, 2), activation='relu')
        self.mp_2   = MaxPooling2D((2, 2))
        self.flat_layer = Flatten()

    def call(self, inputs):
        x = self.conv_0(inputs)
        x = self.mp_0(x)
        x = self.conv_1(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.mp_2(x)
        x = self.flat_layer(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "window_size": self.window_size,
        })
        return config

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
