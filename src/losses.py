import tensorflow as tf


@tf.function
def create_circle_decay_matrix(width, height, power):
    # Calculate the center coordinates
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    # Generate grid points for x and y coordinates
    x = tf.linspace(0, width - 1, width)
    y = tf.linspace(0, height - 1, height)
    x_grid, y_grid = tf.meshgrid(x, y)

    # Calculate the Euclidean distance from each point to the center
    distances = tf.sqrt(tf.square(x_grid - center_x) + tf.square(y_grid - center_y))

    # Calculate the maximum distance from the center
    max_distance = tf.sqrt(tf.square(center_x) + tf.square(center_y))

    # Normalize the distances to the range [0, 1]
    normalized_distances = distances / max_distance

    # Apply a non-linear transformation for faster decay
    decay_matrix = tf.pow(1 - normalized_distances, power)


    return tf.cast(decay_matrix, tf.float32)

def get_companion_std(inputs, prediction):
    residuals = tf.abs(inputs - prediction)
    return tf.math.reduce_sum(residuals)

@tf.function
def keep_back(inputs, prediction, fwhm):
    inputs     = tf.cast(inputs['windows'], tf.float32)
    prediction = tf.cast(prediction, tf.float32)
    residuals  = tf.math.subtract(inputs, prediction)
    
    mask = create_circle_decay_matrix(tf.shape(residuals)[1], tf.shape(residuals)[2], 4)
    
    mask = tf.reshape(mask, [1, tf.shape(residuals)[1], tf.shape(residuals)[2], 1])
    
    res_square = residuals*mask
    return tf.math.reduce_std(res_square) 

