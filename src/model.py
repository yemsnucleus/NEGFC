import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras        import Model

from .layer import ConvBlock, PositionRegressor, TranslationLayer

class CustomModel(Model):
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
            loss = self.loss_fn(y, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=False)
            loss = self.loss_fn(y, y_pred)
        return {'loss': loss}

    @tf.function
    def predict_step(self, data):
        y_pred = self(x, training=False)
        return y_pred

def normalize_batch(batch):
    max_val = tf.reduce_max(batch, axis=[2, 3])
    max_val = tf.expand_dims(max_val, axis=2)
    max_val = tf.expand_dims(max_val, axis=3)
    min_val = tf.reduce_min(batch, axis=[2, 3])
    min_val = tf.expand_dims(min_val, axis=2)
    min_val = tf.expand_dims(min_val, axis=3)
    tensor = (batch-min_val)/(max_val - min_val)
    return tensor

def build_input(window_size):
	inputs = Input(shape=(window_size, window_size, 1),
                         batch_size=None, dtype=tf.float32,
                         name='input')
	return inputs

def create_model(window_size):
    input_placeholder = build_input(window_size)
    conv_block = ConvBlock(window_size, name='convnet')
    pos_regressor = PositionRegressor(name='posreg')
    translate_layer = TranslationLayer(window_size, name='translation')

#     x = normalize_batch(input_placeholder)
    x = conv_block(input_placeholder)
    dx, dy = pos_regressor(x)
    x = translate_layer(input_placeholder, dx, dy)

    return CustomModel(inputs=input_placeholder, outputs=x, name="PosRegressor")