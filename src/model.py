import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras        import Model

from .layer import ConvBlock, PositionRegressor, TranslationLayer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D

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


# ==========================================
# Encoder
def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    return Model(input_layer, encoded)

# Decoder
def build_decoder(encoded_shape):
    input_layer = Input(shape=encoded_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_layer, x)

def create_autoencoder(input_shape):
    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.output.shape[1:])
    input_layer = Input(shape=input_shape)

    encoded = encoder(input_layer)
    decoded = decoder(encoded)
    cropped_decoded = Cropping2D(cropping=((0, 1), (0, 1)))(decoded)  # Crop to (63, 63, 1)

    return CustomModel(input_layer, cropped_decoded, name='autoencoder')









