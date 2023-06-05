import tensorflow as tf


def get_companion_std(inputs, prediction):
	residuals = tf.pow(tf.squeeze(inputs, axis=-1) - prediction, 2)
	return tf.math.reduce_std(residuals)

