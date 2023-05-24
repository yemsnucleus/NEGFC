import tensorflow as tf 
import tensorflow_addons as tfa
import multiprocessing as mp
import numpy as np
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
        loss = tf.pow(tf.math.reduce_std(non_zero_values),2)
    else:
        loss = tf.pow(tf.math.reduce_sum(objetive_reg), 2)
    return loss


@tf.function
def get_rot_coords(width, height, center, radius, rot_angles):
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    width_s  = width/tf.constant(2., dtype=tf.float32)
    height_s = height/tf.constant(2., dtype=tf.float32)

    x_c = center[0] - width_s # origin in the middle of the frame 
    y_c = center[1] - height_s # origin in the middle of the frame

    radius = tf.sqrt(tf.pow(x_c, 2)+tf.pow(y_c, 2))

    angle = tf.atan2(y_c, x_c) # radians
    angle = angle/tf.constant(np.pi)*180.  # degrees
    theta = tf.math.mod(angle, 360.) # bound up to 1 circumference  

    rot_theta = theta - rot_angles # rotate angles 
    rot_theta = tf.experimental.numpy.deg2rad(rot_theta)

    x_s = tf.multiply(radius, tf.cos(rot_theta)) + width_s
    y_s = tf.multiply(radius, tf.sin(rot_theta)) + height_s

    shift_indices = tf.stack([x_s, y_s], axis=2)
    
    return shift_indices

@tf.function
def create_circle_mask(coordinates, w, h, r):
    # Create a grid of coordinates
    grid_x, grid_y = tf.meshgrid(tf.range(w), tf.range(h))

    # Expand dimensions to match the input coordinates
    expanded_grid_x = tf.expand_dims(grid_x, axis=-1)
    expanded_grid_x = tf.cast(expanded_grid_x, tf.float32)
    expanded_grid_y = tf.expand_dims(grid_y, axis=-1)
    expanded_grid_y = tf.cast(expanded_grid_y, tf.float32)

    # Calculate the squared distance from each point to the coordinates
    dist_sq = tf.square(expanded_grid_x - coordinates[:, 0]) + tf.square(expanded_grid_y - coordinates[:, 1])

    # Create the mask by comparing the squared distance to the radius squared
    mask = tf.cast(dist_sq <= r**2, dtype=tf.float32)

    return tf.transpose(mask, [2, 0, 1])    
    

@tf.function
def reduce_std(y_true, y_pred, nfwhm=2, debug=False, minimize='std'):
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
        tf.print(xy_center)
        xy_mask = get_rot_coords(w, h, xy_center, y_true['fwhm']*nfwhm, rot_angles)
        xy_mask = tf.reshape(xy_mask, [n_frames, 2])
        m = create_circle_mask(xy_mask, w, h, y_true['fwhm']*nfwhm)
        m = tf.reshape(m, [n_frames, w, h, 1])
        return m
    
    mask = tf.map_fn(lambda x: fn(x, y_true['rot_angles']), 
                     xycords,                               
                     fn_output_signature=(tf.float32),
                     parallel_iterations=mp.cpu_count()//2)
    
    mask = tf.reshape(mask, [n_candidates, n_frames, w, h, 1])

    objetive_reg = injected*mask
    
    if debug:
        return objetive_reg
    
    if minimize == 'std':
        valid = tf.not_equal(objetive_reg, 0.)
        non_zero_values = tf.boolean_mask(objetive_reg, valid)
        print('std')
        N = tf.cast(tf.shape(non_zero_values)[0], tf.float32)
        loss = tf.math.reduce_std(non_zero_values)*(N-1)
        return tf.cast(loss, tf.float64)
            
    if minimize == 'sum':
        return tf.abs(tf.reduce_sum(objetive_reg))

    
   


