import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras        import Model
from .layer import TranslateCube, PositionRegressor, FluxRegressor, CubeConvBlock, PSFConvBlock, FluxPosRegressor, ConvolutionalLayer


def build_input(window_size):
	inputs = {
		'cube': Input(shape=(80, window_size, window_size, 1),
						 batch_size=None, dtype=tf.float32,
						 name='cube'),
		'psf': Input(shape=(80, window_size, window_size, 1),
					 batch_size=None, dtype=tf.float32,
					 name='psf'),
	}

	return inputs


def create_model(window_size):

    input_placeholder = build_input(window_size)

    cube_lstm = tf.keras.layers.ConvLSTM2D(filters=2, 
                                           kernel_size=(2, 2), 
                                           strides=(1,1),
                                           data_format='channels_last',
                                           return_sequences=True,
                                           name='cube_lstm')

    psf_lstm  = tf.keras.layers.ConvLSTM2D(filters=2, 
                                           kernel_size=(2, 2),
                                           strides=(1,1), 
                                           data_format='channels_last',
                                           return_sequences=True,
                                           name='psf_lstm')
    avg_layer = tf.keras.layers.Average()

    ffn_0 = tf.keras.layers.Dense(256, name='dense_0')
    ffn_1 = tf.keras.layers.Dense(128, name='dense_1')
    ffn_2 = tf.keras.layers.Dense(64, name='dense_2')
    flux_reg = tf.keras.layers.Dense(1, name='flux_reg')
    # dpos_reg = tf.keras.layers.Dense(2, name='dpos_reg')
    # shift_op = TranslateCube(name='shift')

    # ==== flow ===
    output_cube = cube_lstm(input_placeholder['cube'])    
    output_psf  = psf_lstm(input_placeholder['psf'])

    output_cube = tf.reshape(output_cube, 
        [-1, tf.shape(output_cube)[1], tf.reduce_prod(tf.shape(output_cube)[2:])])
    output_psf = tf.reshape(output_psf, 
        [-1, tf.shape(output_psf)[1], tf.reduce_prod(tf.shape(output_psf)[2:])])

    avg_output = avg_layer([output_cube, output_psf])

    out_0 = ffn_0(avg_output)
    out_1 = ffn_1(out_0)
    out_2 = ffn_2(out_1)

    pred_flux = flux_reg(out_2)
    # pred_dpos = dpos_reg(out_2)
    # dx = tf.slice(pred_dpos, [0, 0, 0], [-1, -1, 1], name='dx')
    # dy = tf.slice(pred_dpos, [0, 0, 1], [-1, -1, 1], name='dy') 

    # shift_fake = shift_op((input_placeholder['psf'], dx, dy, window_size))
    final_fake = input_placeholder['psf']*tf.reshape(pred_flux, [tf.shape(pred_flux)[0], tf.shape(pred_flux)[1], 1, 1, 1])

    return CustomModel(inputs=input_placeholder, outputs=final_fake, name="convlstm")


def cut_graph_to_layer(model, layer_name):
  """Cuts the graph of a TensorFlow model to a certain layer."""

  
  
  return new_model


class CustomModel(tf.keras.Model):
    '''
    Custom functional model
    '''
    def compile(self, loss_fn, **kwargs):
        super(CustomModel, self).compile(**kwargs)
        self.loss_fn = loss_fn
        
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.loss_fn(data, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data, training=False)
            loss = self.loss_fn(data, y_pred)
        return {'loss': loss}

    @tf.function
    def predict_step(self, data):
        # y_pred = self(data)
        cube_lstm = self.get_layer('cube_lstm')
        psf_lstm  = self.get_layer('psf_lstm')
        ffn_0   = self.get_layer('dense_0')
        ffn_1   = self.get_layer('dense_1')
        ffn_2   = self.get_layer('dense_2')
        flux_reg  = self.get_layer('flux_reg')
        avg_layer = self.get_layer('average')

        output_cube = cube_lstm(data['cube'])    
        output_psf  = psf_lstm(data['psf'])

        output_cube = tf.reshape(output_cube, 
            [-1, tf.shape(output_cube)[1], tf.reduce_prod(tf.shape(output_cube)[2:])])
        output_psf = tf.reshape(output_psf, 
            [-1, tf.shape(output_psf)[1], tf.reduce_prod(tf.shape(output_psf)[2:])])

        avg_output = avg_layer([output_cube, output_psf])

        out_0 = ffn_0(avg_output)
        out_1 = ffn_1(out_0)
        out_2 = ffn_2(out_1)

        pred_flux = flux_reg(out_2)

        return pred_flux