import tensorflow as tf

from .fake_comp import create_patch

def format_input(cube, psf, rot):
    return {
        'cube':cube,
        'psf':psf,
        'rot_angles':rot
    }, None

def get_dataset(cube, psf, rot_ang, lambda_ch=0, psf_pos=0):    
    psf = create_patch(cube[lambda_ch, psf_pos], psf[lambda_ch])
    
    min_val = tf.expand_dims(tf.expand_dims(tf.reduce_min(cube[lambda_ch], axis=[1, 2]), 1), 2)
    max_val = tf.expand_dims(tf.expand_dims(tf.reduce_max(cube[lambda_ch], axis=[1, 2]), 1), 2)

    normed_cube = (cube[lambda_ch] - min_val)/(max_val - min_val)  
    
    
    dataset = tf.data.Dataset.from_tensor_slices((normed_cube[None,...], 
                                                  psf[None,...], 
                                                  rot_ang[None,...]))
    dataset = dataset.map(format_input)
    return dataset.batch(1), (min_val, max_val)
 