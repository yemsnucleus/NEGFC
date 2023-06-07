import tensorflow as tf


def get_companion_std(inputs, prediction):
	residuals = tf.abs(tf.squeeze(inputs, axis=-1) - prediction)
	return tf.math.reduce_sum(residuals)

def keep_back(inputs, prediction):
    inputs     = tf.cast(inputs, tf.float64)
    prediction = tf.cast(prediction, tf.float64)
    windows    = tf.squeeze(inputs, axis=-1)
    residuals  = tf.math.subtract(windows, prediction)
    res_square = residuals
    return tf.pow(tf.math.reduce_std(res_square), 2) 

