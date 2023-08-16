import tensorflow as tf


def create_circle_mask(image_size, radius, center):
	"""
	Create a circle mask inside a 2D image.

	Args:
	    image_size (int): Size of the image (assumes square image).
	    radius (int): Radius of the circle.
	    center (tuple): Tuple specifying the center coordinates of the circle (x, y).

	Returns:
	    tf.Tensor: Circle mask tensor with shape [image_size, image_size, 1].
	"""

	# Create meshgrid
	xx, yy = tf.meshgrid(tf.range(image_size), tf.range(image_size))
	xx = tf.cast(xx, tf.float32)
	yy = tf.cast(yy, tf.float32)

	# Calculate distance from center
	distance = tf.sqrt(tf.square(xx - tf.cast(center[0], tf.float32)) + tf.square(yy - tf.cast(center[1], tf.float32)))

	# Create circle mask
	mask = tf.cast(distance <= tf.cast(radius, tf.float32), tf.float32)

	# Expand dimensions for broadcasting
	mask = tf.expand_dims(mask, axis=-1)

	return mask

def reduce_moments(y_true, y_pred, moments, fwhm=None, debug=False):	
	back_mean, back_std = tf.split(moments, 2, axis=1)
	residuals = tf.pow(y_true - y_pred, 2)
	

	if fwhm is not None:
		out_shp = tf.shape(y_true)

		outter_mask = create_circle_mask(out_shp[2], 
										 radius=2*fwhm/2, 
										 center=(out_shp[2]/2, out_shp[2]/2))
		residuals = residuals * tf.reshape(outter_mask, [1, 1, out_shp[2], out_shp[3], 1])			   


	if debug:
		return residuals

	non_zero_indices = tf.where(tf.not_equal(residuals, 0.))
	residuals = tf.gather_nd(residuals, non_zero_indices)

	res_std = tf.math.reduce_std(residuals)
	res_mean = tf.math.reduce_mean(residuals)

	std = tf.cast(back_std, tf.float32) - res_std
	std = tf.pow(std, 2)
	
	mean = tf.cast(back_mean, tf.float32) - res_mean
	mean = tf.pow(mean, 2)

	return tf.reduce_mean(mean), tf.reduce_mean(std)