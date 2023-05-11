import tensorflow as tf 

from .fake_comp import create_circular_mask

@tf.function
def custom_loss(adi_fake, radius, rot_theta, fwhm=4):
    w = tf.shape(adi_fake)[-2]
    h = tf.shape(adi_fake)[-1]
    
    w_s = tf.cast(w, tf.float32)/tf.constant(2., dtype=tf.float32)
    h_s = tf.cast(h, tf.float32)/tf.constant(2., dtype=tf.float32)
    
    x_pos = tf.multiply(radius, tf.cos(rot_theta)) + w_s
    y_pos = tf.multiply(radius, tf.sin(rot_theta)) + h_s

    mask = create_circular_mask(w, h, center=(x_pos, y_pos), radius=fwhm)
    
    objetive_reg =  adi_fake * mask
    objetive_reg = tf.reshape(objetive_reg, [w*h])   
    # get a boolean mask for non-zero values
    mask = tf.not_equal(objetive_reg, 0.)
    # use the mask to get non-zero values
    non_zero_values = tf.boolean_mask(objetive_reg, mask)
    
    abs_std = tf.abs(tf.math.reduce_std(non_zero_values))
    return abs_std