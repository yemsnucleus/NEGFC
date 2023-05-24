import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf
import multiprocessing
import numpy as np
import pandas as pd

from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm, sigma_clipped_stats
from vip_hci.metrics.snr_source     import snr
from skimage.feature				import peak_local_max
from vip_hci.preproc.derotation     import cube_derotate

try:
    from photutils.aperture import aperture_photometry, CircularAperture
except:
    from photutils import aperture_photometry, CircularAperture

@tf.function
def gauss_model(params, mean, scale, amplitude):
    f = tf.exp(-((params[:, 0]-mean[0])**2/(2*scale[0]**2) + (params[:, 1]-mean[1])**2/(2*scale[1]**2)))
    return amplitude*f

def adjust_gaussian(frame, return_history=False, n_iters=50, learning_rate=1e-1, init_x=None, init_y=None, init_scale=2):
    tpsf      = tf.convert_to_tensor(frame)
    indices   = tf.cast(tf.where(tpsf), tf.float32)
    shp_tpsf  = tf.shape(tpsf)
    
    if init_x is None:
        init_x    = tf.cast(shp_tpsf[0]//2, tf.float32)
        
    if init_y is None:
        init_y    = tf.cast(shp_tpsf[1]//2, tf.float32)
            
    flat_tpsf = tf.reshape(tpsf, [-1])
    
    amplitude  = tf.reduce_max(flat_tpsf)
    init_x     = tf.cast(init_x, tf.float32)
    init_y     = tf.cast(init_y, tf.float32)
    init_scale = tf.cast(init_scale, tf.float32)
    
    mean       = tf.Variable([init_x, init_y])
    scale      = tf.Variable([init_scale, init_scale])
    
    optimizer = tf.optimizers.Adam(learning_rate)

    losses = []
    for i in range(n_iters):
        with tf.GradientTape() as tape:
            y_pred = gauss_model(indices, mean, scale, amplitude)
            loss = tf.keras.metrics.mean_squared_error(flat_tpsf, y_pred)
            losses.append(loss)
            # Compute gradients
            trainable_vars = [mean, scale]
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            optimizer.apply_gradients(zip(gradients,trainable_vars))
    
    response = {
        'mean': mean, 
        'scale': scale,
        'fwhm': scale*gaussian_sigma_to_fwhm,
        'amplitude': amplitude,
        'history': losses,
    }
    
    return response

def center_frame(frame, fwhm):
    x_ref = tf.cast(tf.shape(frame)[-1]//2, tf.float32)
    y_ref = tf.cast(tf.shape(frame)[-2]//2, tf.float32)
    gauss_res = adjust_gaussian(frame)
    shift_x = x_ref - gauss_res['mean'][0]
    shift_y = y_ref - gauss_res['mean'][1] 
    
    return tfa.image.translate(frame, [shift_x, shift_y])    
    
def center_cube(cube, fwhm):
    return tf.map_fn(lambda x: center_frame(x, fwhm), cube)

def normalize_psf(psf, fwhm):
    fwhm_radius = fwhm / 2.0
    aperture_radius = tf.cast(fwhm_radius * tf.sqrt(2.0), tf.int32)

    psf_tensor = tf.transpose(psf, [1, 2, 0])

    aperture_mask = tf.ones_like(psf_tensor)
    aperture_mask = tf.image.crop_to_bounding_box(psf_tensor, 
                                                  tf.shape(psf_tensor)[0] // 2 - aperture_radius, 
                                                  tf.shape(psf_tensor)[1] // 2 - aperture_radius, 
                                                  aperture_radius * 2, 
                                                  aperture_radius * 2)

    aperture_mask = tf.cast(aperture_mask, dtype=tf.float32)
    aperture_flux = tf.reduce_sum(aperture_mask, axis=[0, 1])

    psf_tensor = psf_tensor/aperture_flux
    return tf.transpose(psf_tensor, [2, 0, 1])

def create_patch(frame, template):
    w = tf.shape(template)[-1]//2
    cenx, ceny = tf.shape(frame)[-2]//2, tf.shape(frame)[-1]//2
    start_y, end_y = int(ceny-w), int(ceny+w+1)
    start_x, end_x = int(cenx-w), int(cenx+w+1)

    frame_copy = tf.zeros_like(frame)
    indices = tf.stack(tf.meshgrid(tf.range(start_y, end_y), tf.range(start_x, end_x)), axis=-1)
    template_reshape = tf.reshape(template, [-1, template.shape[-1]])

    frame_copy = tf.tensor_scatter_nd_update(frame_copy, indices, template_reshape)
    return frame_copy

def inject_fake(x, y, flux, cube, patch, rot_angles):
    x_center, y_center = cube.shape[1]/2, cube.shape[2]/2
    x = tf.cast(x, tf.float32) - x_center
    y = tf.cast(y, tf.float32) - y_center
    flux = tf.cast(flux, tf.float32)
    
    cube_patch = tf.expand_dims(patch, 0)
    cube_patch = tf.tile(cube_patch, [cube.shape[0], 1, 1])
    cube_patch = tf.expand_dims(cube_patch, -1)
    
    pi = tf.constant(np.pi)

    angle = tf.atan2(y, x) 
    angle = angle/pi*180.
    theta = tf.math.mod(angle, 360.) 

    radius = tf.sqrt(tf.pow(x, 2)+tf.pow(y, 2))
    shifted_theta = theta - rot_angles
    shifted_theta = tf.experimental.numpy.deg2rad(shifted_theta)
    
    x_s = tf.multiply(radius, tf.cos(shifted_theta))
    y_s = tf.multiply(radius, tf.sin(shifted_theta))
    x_s = tf.expand_dims(x_s,-1)
    y_s = tf.expand_dims(y_s,-1)
    
    shift_indices = tf.stack([x_s, y_s], axis=1)
    shift_indices = tf.squeeze(shift_indices)
    
    fake_comp = tfa.image.translate(cube_patch*flux, shift_indices)
    fake_comp = tf.squeeze(fake_comp)
    
    return cube + fake_comp

@tf.function
def pca_tf(cube, out_size, ncomp=1):
    nframes = out_size[0]
    height = out_size[1]
    width = out_size[2]
    
    data = tf.reshape(cube, [nframes, height*width])
    data_mean = tf.reduce_mean(data, 1)
    data_centered = data - tf.expand_dims(data_mean, 1)
    data_centered = tf.reshape(data_centered, [nframes, height*width])
    s, u, v = tf.linalg.svd(data_centered)
    U =tf.transpose(u[:, :ncomp])
    transformed   = tf.matmul(U, data_centered)
    reconstructed = tf.transpose(tf.matmul(tf.transpose(transformed), U))
    residuals     = data_centered - reconstructed
    residuals_i   = tf.reshape(residuals, [nframes, height, width])
    reconstructed_i = tf.reshape(reconstructed, [nframes, height, width])

    residuals_i.set_shape([nframes, height, width])
    reconstructed_i.set_shape([nframes, height, width])
    
    return residuals_i, reconstructed_i
# @tf.function
# def pca_tf(cube, ncomp=1):
#     data = tf.reshape(cube, [tf.shape(cube)[0], tf.shape(cube)[1]*tf.shape(cube)[2]])
#     data_mean = tf.reduce_mean(data, 1)
#     data_centered = data - tf.expand_dims(data_mean, 1)
#     data_centered = tf.reshape(data_centered, 
#                                [tf.shape(cube)[0], 
#                                 tf.shape(cube)[1]*tf.shape(cube)[2]])
#     data_centered = tf.Tensor.set_shape([tf.shape(cube)[0], tf.shape(cube)[1]*tf.shape(cube)[2]])
#     s, u, v = tf.linalg.svd(data_centered)
#     U =tf.transpose(u[:, :ncomp])
#     transformed   = tf.matmul(U, data_centered)
#     reconstructed = tf.transpose(tf.matmul(tf.transpose(transformed), U))
#     residuals     = data_centered - reconstructed
#     residuals_i   = tf.reshape(residuals, [-1, tf.shape(cube)[1], tf.shape(cube)[2]])
#     reconstructed_i = tf.reshape(reconstructed, [-1, tf.shape(cube)[1], tf.shape(cube)[2]])
#     return residuals_i, reconstructed_i

def rotate_cube(cube, rot_ang, derotate='tf', verbose=0):

    if derotate=='tf':
        if verbose: print('Using Tensorflow on GPU')
        
        rot_ang_deg = tf.experimental.numpy.deg2rad(rot_ang)
        if tf.rank(cube) == 3:
            cube = tf.expand_dims(cube, -1)
            
        res_derot = tfa.image.rotate(cube, -rot_ang_deg, 
                                     interpolation='nearest', 
                                     fill_mode='reflect')

    else:
        cores = multiprocessing.cpu_count()-2
        if verbose: print(f'Using VIP on {cores} cores')
        res_derot = cube_derotate(cube, 
                                  rot_ang, 
                                  nproc=cores, 
                                  imlib='vip-fft', 
                                  interpolation='nearneig')
        
    return res_derot

def apply_adi(cube, rot_ang, out_size, ncomp=1, derotate='tf', return_cube=False):
    res, cube_rec = pca_tf(cube, out_size, ncomp=ncomp)
    res_derot = rotate_cube(res, rot_ang, derotate=derotate)
    res_derot = tf.reshape(res_derot, [tf.shape(cube)[0], tf.shape(cube)[1], tf.shape(cube)[2]])
    median = tfp.stats.percentile(res_derot, 50.0, 
                                  interpolation='midpoint', axis=0)
    median = tf.reshape(median, [tf.shape(cube)[1], tf.shape(cube)[2]])
    
    if return_cube:
        return median, res_derot
    return median

def get_coords(adi_image, fwhm=4, bkg_sigma = 5, cut_size = 10, num_peaks=20):
    _, median, stddev = sigma_clipped_stats(adi_image, sigma=bkg_sigma, maxiters=None)
    bkg_level = median + (stddev * bkg_sigma)

    coords_temp = peak_local_max(adi_image, threshold_abs=bkg_level,
                                 min_distance=int(np.ceil(fwhm)),
                                 num_peaks=num_peaks)

    coords, fluxes, fwhm_mean, snr_list = [], [], [], []
    table = pd.DataFrame()
    for y_pos, x_pos in coords_temp:

        bbox = [int(y_pos+cut_size/2.), int(x_pos+cut_size/2.), 
                int(y_pos-cut_size/2.), int(x_pos-cut_size/2.)]
        bbox = tf.expand_dims(bbox, 0)

        # Define the target output size
        target_size = [cut_size, cut_size]

        # Cut the square from the input image
        off_y = tf.cast(y_pos-cut_size//2, tf.int32)
        off_x = tf.cast(x_pos-cut_size//2, tf.int32)

        cropped_image = tf.image.crop_to_bounding_box(tf.expand_dims(adi_image, -1), 
                                                      off_y, off_x, cut_size, cut_size)

        results = adjust_gaussian(cropped_image, 
                                  init_x=x_pos-off_x, 
                                  init_y=y_pos-off_y,
                                  init_scale=fwhm)

        xxc = results['mean'][0]+tf.cast(off_x, tf.float32)
        yyc = results['mean'][1]+tf.cast(off_y, tf.float32)

        fwhm_y = results['scale'][0] * gaussian_sigma_to_fwhm
        fwhm_x = results['scale'][1] * gaussian_sigma_to_fwhm

        mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)])
        # Filtering Process
        condyf = np.allclose(yyc, y_pos, atol=2)
        condxf = np.allclose(xxc, x_pos, atol=2)
        condmf = np.allclose(mean_fwhm_fit, fwhm, atol=3)
        
        try:
            val_snr = snr(adi_image, (xxc, yyc), 
                          mean_fwhm_fit, False, verbose=False)
        except Exception as e:
            val_snr = None
            
        aper = CircularAperture((x_pos, y_pos), r=mean_fwhm_fit / 2.) 
        obj_flux_i = aperture_photometry(adi_image, aper, method='exact')
        obj_flux_i = obj_flux_i['aperture_sum'][0]
            
        if results['amplitude'] > 0 and condxf and condyf and condmf and val_snr is not None:
            coords.append((yyc, xxc))
            fluxes.append(obj_flux_i)
            fwhm_mean.append(mean_fwhm_fit)
            snr_list.append(val_snr)

    coords = np.array(coords)
    table['x']    = coords[:, 1]
    table['y']    = coords[:, 0]
    table['flux'] = fluxes
    table['fwhm_mean'] = fwhm_mean
    table['snr']  = snr_list
    return table

def create_circular_mask(height, width, center=None, radius=None):
    if center is None:
        center = (height//2, width//2)
    if radius is None:
        radius = min(center[0], center[1], height-center[0], width-center[1])
        
    Y, X = tf.meshgrid(tf.range(height), tf.range(width))
    height = tf.cast(height, tf.float32)
    width  = tf.cast(width, tf.float32)
 
    Y = tf.cast(Y, tf.float32)
    X = tf.cast(X, tf.float32)
    center = (tf.cast(center[0], tf.float32), tf.cast(center[1], tf.float32))
    radius = tf.cast(radius, tf.float32)
    
    dist_from_center = tf.linalg.norm(tf.stack([tf.cast(Y - center[0], tf.float32), 
                                                tf.cast(X - center[1], tf.float32)], axis=-1), axis=-1)
    dist_from_center = tf.cast(dist_from_center, tf.float32)
    mask = tf.where(dist_from_center <= radius, 1.0, 0.0)
    return mask

def get_objective_region(cube, x, y, rot_ang, fwhm):
    
    w = tf.shape(cube)[-2]
    h = tf.shape(cube)[-1] 
    
    mask = create_circular_mask(w, h, center=(x, y), radius=fwhm)
    mask = tf.expand_dims(mask, 0)
    
    
    if tf.rank(cube) > 2:
        nframes = tf.shape(cube)[0]
        mask = tf.tile(mask, [nframes, 1, 1])
        cube_rot = rotate_cube(cube, rot_ang=rot_ang, derotate='tf')
        cube_rot = tf.squeeze(cube_rot)
        objetive_reg =  cube_rot * mask
        objetive_reg = tf.reshape(objetive_reg, [nframes, w, h])   
    else:
        cube_rot = cube
        objetive_reg =  cube_rot * mask

    return  objetive_reg   