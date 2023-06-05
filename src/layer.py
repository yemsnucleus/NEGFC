import tensorflow as tf

from tensorflow.keras.layers import Input, Layer
import tensorflow_addons as tfa

@tf.function
def translate_image(inputs):
	inp_tensor, dx, dy, winsize = inputs
	cords = tf.stack([dx, dy], 1)
	inp_tensor = tf.squeeze(inp_tensor, -1)
	cords = tf.squeeze(cords)
	out = tfa.image.translate(inp_tensor, cords)

	# ---- OLD CODE ---
	# inp_tensor = tf.reshape(inp_tensor, [winsize, winsize, 1])
	# indices = tf.where(inp_tensor)
	# indices = tf.cast(indices, tf.int32)
	# indices_x = tf.slice(indices, [0,0], [-1, 1])
	# indices_y = tf.slice(indices, [0,1], [-1, 1])
	# indices_x_shift = indices_x + tf.cast(dx, tf.int32)
	# indices_y_shift = indices_y + tf.cast(dy, tf.int32)
	# indices_x_shift = tf.squeeze(indices_x_shift)
	# indices_y_shift = tf.squeeze(indices_y_shift)

	# mask_x  = tf.where(indices_x_shift < winsize, True, False)
	# mask_y  = tf.where(indices_y_shift < winsize, True, False)
	# mask = tf.math.logical_and(mask_x, mask_y)
	# mask = tf.reshape(mask, [tf.shape(inp_tensor)[0]*tf.shape(inp_tensor)[1]])
	# indices_x_shift = tf.boolean_mask(indices_x_shift, mask)
	# indices_y_shift = tf.boolean_mask(indices_y_shift, mask)
	# shift_indices = tf.stack([indices_x_shift, indices_y_shift], 1)
	
	# updates = tf.reshape(inp_tensor, [tf.shape(inp_tensor)[0]*tf.shape(inp_tensor)[1]])
	# updates = tf.boolean_mask(updates, mask)

	# tensor = tf.zeros_like(inp_tensor)
	# out = tf.tensor_scatter_nd_update(tf.squeeze(tensor), shift_indices, updates)
	return out

class TranslateCube(Layer):
	def __init__(self):
		super(TranslateCube, self).__init__()

	def call(self, inputs):
		images = inputs[0]
		dx = inputs[1]
		dy = inputs[2]
		winsize = inputs[3]
		winsize = tf.tile(tf.expand_dims(winsize, 0), [tf.shape(dx)[0]])

		x = tf.map_fn(lambda x: translate_image(x), (images, dx, dy, winsize), fn_output_signature=tf.float32)
		return x

