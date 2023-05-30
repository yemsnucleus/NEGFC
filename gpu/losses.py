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
    r = tf.cast(r, tf.float32)
    grid_x, grid_y = tf.meshgrid(tf.range(w), tf.range(h))
    tf.print(grid_x)
    # Expand dimensions to match the input coordinates
    expanded_grid_x = tf.cast(grid_x, tf.float32)
    expanded_grid_y = tf.cast(grid_y, tf.float32)
    x_cords = tf.expand_dims(tf.expand_dims(coordinates[:, 0], -1), -1)
    tf.print(x_cords)
    y_cords = tf.expand_dims(tf.expand_dims(coordinates[:, 1], -1), -1)
    # Calculate the squared distance from each point to the coordinates
    dist_sq = tf.square(tf.pow(expanded_grid_x - x_cords, 2) + \
                        tf.pow(expanded_grid_y - y_cords, 2))
    # Create the mask by comparing the squared distance to the radius squared
    mask = tf.cast(dist_sq <= r, dtype=tf.float32)
    return mask
    

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

    
def create_circle_cube(pairs, radius, dimensions):
    mask = tf.map_fn(lambda x: create_circle_mask_2(dimensions, x, radius),
                     pairs,
                     fn_output_signature=(tf.float32),
                     parallel_iterations=mp.cpu_count()//2)
    # Reshape the cube to (N, W, H, 1)
    mask = tf.reshape(mask, (-1, dimensions[0], dimensions[1], 1))
    return mask

def create_circle_mask_2(image_shape, center, radius):
    # Create a meshgrid of coordinates
    x_coords, y_coords = tf.meshgrid(tf.range(image_shape[1]), tf.range(image_shape[0]))
    x_coords = tf.cast(x_coords, tf.float32)
    y_coords = tf.cast(y_coords, tf.float32)
    # Calculate the distance of each coordinate from the center
    distances = tf.sqrt(tf.square(x_coords - center[0]) + tf.square(y_coords - center[1]))

    # Create a boolean mask where the distances are less than the radius
    mask = tf.cast(distances <= radius, tf.float32)

    return mask

@tf.function
def reduce_std_angle(y_true, y_pred, nfwhm=2, debug=False, minimize='std'):
    fake   = y_pred[0]
    radius = y_pred[1]
    theta  = y_pred[2]

    n_candidates = tf.shape(fake)[0]
    n_frames = tf.shape(fake)[1]
    w = tf.shape(fake)[2]
    h = tf.shape(fake)[3]

    cube = tf.tile(y_true['cube'], [n_candidates, 1, 1, 1])
    cube = tf.expand_dims(cube, -1)

    injected = fake + cube
    
    # Create a grid of x and y coordinates
    x_coords = tf.range(w, dtype=tf.float32)
    y_coords = tf.range(h, dtype=tf.float32)
    X, Y = tf.meshgrid(x_coords, y_coords)
    Xf = tf.tile(tf.expand_dims(X, 0), [n_frames, 1, 1])
    Yf = tf.tile(tf.expand_dims(Y, 0), [n_frames, 1, 1])
    
    def fn(currr, currtheta, currfwhm):
        rot_theta = currtheta - y_true['rot_angles']# rotate angles 
        rot_theta = tf.experimental.numpy.deg2rad(rot_theta)
        x0 = tf.multiply(currr, tf.cos(rot_theta)) + tf.cast(h/2, tf.float32)
        y0 = tf.multiply(currr, tf.sin(rot_theta)) + tf.cast(w/2, tf.float32)   
        x0 = tf.reshape(x0, [-1, 1, 1])
        y0 = tf.reshape(y0, [-1, 1, 1])

        distances = tf.sqrt(tf.pow(Xf - x0, 2) + tf.pow(Yf - y0, 2))
        mask = tf.where(distances <= nfwhm*currfwhm, 1., 0.)
        return mask
        
    mask = tf.map_fn(lambda x: fn(x[0], x[1], x[2]),
                     (radius, theta, y_true['fwhm'][0]),
                     fn_output_signature=(tf.float32))

    mask = tf.expand_dims(mask, -1)

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
   


