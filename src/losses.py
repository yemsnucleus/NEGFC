import tensorflow as tf


@tf.function
def create_circle_decay_matrix(width, height, power, dx=None, dy=None):
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

@tf.function
def create_circle_mask(width, height, radii, decay_factor=2, xc=None, yc=None):
    x = tf.range(width,
                 dtype=tf.float32)
    y = tf.range(height, dtype=tf.float32)
    xx, yy = tf.meshgrid(x, y)
    
    xx = tf.cast(xx, tf.float32)
    yy = tf.cast(yy, tf.float32)
    
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
       
    xc =  width // 2 + xc
    yc =  height // 2 + yc
 
    def process_radius(radius, xcc, ycc):
        radius = tf.cast(radius, tf.float32)
        distance = tf.sqrt(tf.square(xx - xcc) + tf.square(yy - ycc))
        decayed_distance = tf.pow(distance / radius, decay_factor)
        mask = 1.0 - tf.clip_by_value(decayed_distance, 0.0, 1.0)
        return mask
        
    masks = tf.map_fn(lambda x: process_radius(radius=x[0], 
                                               xcc=x[1], 
                                               ycc=x[2]), 
                      (radii, xc, yc), 
                      fn_output_signature=tf.float32)
    masks = tf.expand_dims(masks, -1)
    return masks

def get_companion_std(inputs, prediction):
    residuals = tf.abs(inputs - prediction)
    return tf.math.reduce_sum(residuals)

def wrapper(fn, **kwargs):
    def inner(*args):
        out = fn(*args, **kwargs)
        return out
    return inner

@tf.function
def keep_back(inputs, outputs, params, prediction, decay_factor=2.):
    inp_windows = tf.cast(inputs['windows'], tf.float32)
    
    mask   = create_circle_mask(tf.shape(inp_windows)[1], tf.shape(inp_windows)[2], 
                                outputs['fwhm'], decay_factor=decay_factor, xc=params[0], yc=params[1])
    
    prediction = tf.cast(prediction, tf.float32)
    residuals  = tf.math.subtract(inp_windows, prediction) * mask
    res_term = tf.reduce_mean(tf.pow(residuals, 2))
        
    backres = tf.math.subtract(residuals, outputs['windows'])
    back_term = tf.reduce_mean(tf.pow(backres, 2)) 
    
    return 0.3*res_term + 0.7*back_term

@tf.function
def reduce_std(data, y_pred):
    y_true = data['cube']

    residuals = y_true - y_pred

    return tf.math.reduce_std(residuals)