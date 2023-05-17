import tensorflow as tf 
import tensorflow_addons as tfa
import multiprocessing as mp
from .fake_comp import create_circular_mask, rotate_cube

@tf.function
def custom_loss(adi_fake, radius, rot_theta, fwhm=4, std=True):
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
def reduce_std(y_true, y_pred):
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
        m = create_circular_mask(w, h, center=xy_center, radius=4)
        m = tf.expand_dims(m, 0)
        m = tf.tile(m, [tf.shape(rot_angles)[1], 1, 1]) 
        m = tf.expand_dims(m, -1)
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

    return tf.math.reduce_std(non_zero_values)


