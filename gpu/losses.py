import tensorflow as tf 
import tensorflow_addons as tfa
import multiprocessing as mp
from .fake_comp import create_circular_mask, rotate_cube

def wrapper(fn, **kwargs):
    def inner(*args):
        out = fn(*args, **kwargs)
        return out
    return inner

@tf.function
def custom_loss(adi_fake, radius, rot_theta, fwhm=3, std=True):
    w = tf.shape(adi_fake)[-2]
    h = tf.shape(adi_fake)[-1]
    
    w_s = tf.cast(w, tf.float32)/tf.constant(2., dtype=tf.float32)
    h_s = tf.cast(h, tf.float32)/tf.constant(2., dtype=tf.float32)
    
    x_pos = tf.multiply(radius, tf.cos(rot_theta)) + w_s
    y_pos = tf.multiply(radius, tf.sin(rot_theta)) + h_s

    mask = create_circular_mask(w, h, center=(x_pos, y_pos), radius=fwhm)
    
    objetive_reg =  adi_fake * mask
    objetive_reg = tf.reshape(objetive_reg, [w*h])   
    
    if std:
        # get a boolean mask for non-zero values
        mask = tf.not_equal(objetive_reg, 0.)
        # use the mask to get non-zero values
        non_zero_values = tf.boolean_mask(objetive_reg, mask)
        loss = tf.pow(tf.math.reduce_std(non_zero_values), 2)
    else:
        loss = tf.pow(tf.math.reduce_sum(objetive_reg), 2)
    return loss


@tf.function
def apply_circle_mask(n_frames, height, width, coordinates, radius):
    matrices = tf.ones([n_frames, height, width], dtype=tf.float32)

    # Generate meshgrid of indices
    y_indices, x_indices = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
    y_indices = tf.expand_dims(y_indices, axis=-1)
    x_indices = tf.expand_dims(x_indices, axis=-1)
    y_indices = tf.cast(y_indices, tf.float32)
    x_indices = tf.cast(x_indices, tf.float32)
    # Compute distance from each coordinate
    
    distances = tf.square(x_indices - coordinates[:, 0]) + tf.square(y_indices - coordinates[:, 1])

    # Create circular masks
    pow_radius = tf.cast(tf.pow(radius, 2), tf.float32)
    distances = tf.cast(distances, tf.float32)
    masks = tf.cast(distances <= pow_radius, matrices.dtype)
    # Apply masks to matrices
    masked_matrices = matrices * tf.transpose(masks, [2,0,1])
    return masked_matrices

@tf.function
def reduce_std(y_true, y_pred, radius=2, debug=False):
    fake = y_pred[0]
    xycords = y_pred[1]
    
    n_candidates = tf.shape(fake)[0]
    n_frames = tf.shape(fake)[1]
    w = tf.shape(fake)[2]
    h = tf.shape(fake)[3]

    cube = tf.tile(y_true['cube'], [n_candidates, 1, 1, 1])
    cube = tf.expand_dims(cube, -1)

    injected = fake + cube
    
    def fn(xy_center, rot_angles):
        m = apply_circle_mask(n_frames, w, h, xy_center, radius)
        m = tf.reshape(m, [n_frames, w, h, 1])
        rot_angles = tf.reshape(rot_angles, [tf.shape(rot_angles)[1]])
        
        m_derot = tfa.image.rotate(m, -rot_angles, 
                                   interpolation='nearest', 
                                   fill_mode='reflect')

        m_derot = tf.reshape(m_derot, [n_frames, w, h, 1])
        
        return m_derot
    
    mask = tf.map_fn(lambda x: fn(x, y_true['rot_angles']), 
                     xycords,                               
                     fn_output_signature=(tf.float32),
                     parallel_iterations=mp.cpu_count()//2)
    
        
    objetive_reg = injected*mask
    valid = tf.not_equal(objetive_reg, 0.)
    non_zero_values = tf.boolean_mask(objetive_reg, valid)
    
    if debug:
        return objetive_reg
    
    return tf.math.reduce_std(non_zero_values)


