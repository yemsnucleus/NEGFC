import tensorflow as tf


def reduce_moments(y_true, y_pred, moments):	
	back_mean, back_std = tf.split(moments, 2, axis=1)
	residuals = tf.pow(y_true - y_pred, 2)

	res_std = tf.math.reduce_std(residuals, axis=[2, 3, 4])
	res_std = tf.transpose(res_std)

	res_mean = tf.math.reduce_mean(residuals, axis=[2, 3, 4])
	res_mean = tf.transpose(res_mean)

	std = tf.cast(back_std, tf.float32) - res_std
	std = tf.pow(std, 2)
	
	mean = tf.cast(back_mean, tf.float32) - res_mean
	mean = tf.pow(mean, 2)

	return tf.reduce_mean(mean), tf.reduce_mean(std)