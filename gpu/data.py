import tensorflow as tf

from .fake_comp import create_patch

def format_input(cube, psf, rot):
    return {
        'cube':cube,
        'psf':psf,
        'rot_angles':rot
    }, None

def get_dataset(cube, psf, rot_ang, lambda_ch=0, psf_pos=0, normalize=0):    
    psf = create_patch(cube[lambda_ch, psf_pos], psf[lambda_ch])
    
    if normalize == 0:
        cube_inp = cube[lambda_ch]
        recovery = None
        
    if normalize == 1:
        print('MIN MAX SCALER')
        min_val = tf.reduce_min(cube[lambda_ch])
        max_val = tf.reduce_max(cube[lambda_ch])
        cube_inp = (cube[lambda_ch] - min_val)/(max_val - min_val)
        recovery = (min_val, max_val)

    if normalize == 2:
        print('Z-SCORE')
        mean_val = tf.math.reduce_mean(cube[lambda_ch])
        std_val = tf.math.reduce_std(cube[lambda_ch])
        cube_inp = (cube[lambda_ch] - mean_val)/std_val
        recovery = (mean_val, std_val)

    dataset = tf.data.Dataset.from_tensor_slices((cube_inp[None,...], 
                                                  psf[None,...], 
                                                  rot_ang[None,...]))
    dataset = dataset.map(format_input)
    
    return dataset.batch(1), recovery
 