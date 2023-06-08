import tensorflow as tf
import numpy as np

DTYPE = tf.float32
DTYPEINT = tf.int32
def reshape_image(image, new_shape):
    # Get the current dimensions of the image
    current_height, current_width = tf.shape(image)[-2], tf.shape(image)[-1]

    # Calculate the padding or cropping values
    pad_height = (new_shape - current_height) // 2
    pad_width = (new_shape - current_width) // 2

    # If the new shape is larger, pad the image
    if new_shape >= current_height and new_shape >= current_width:
        paddings = [[pad_height, pad_height], [pad_width, pad_width]]
        image = tf.pad(image, paddings)
    
    # If the new shape is smaller, crop the image
    image = tf.cast(image, dtype=DTYPE)
    return tf.slice(image, [current_height//2-new_shape//2, current_width//2-new_shape//2], [new_shape, new_shape])

def get_rotated_coords(x_pos, y_pos, cube, rot_angles):
	# Getting rotated coords
	nframes, height, width = cube.shape
	width = tf.cast(width, DTYPE)
	height = tf.cast(height, DTYPE)
	width_mid  = width/tf.constant(2., dtype=DTYPE)
	height_mid = height/tf.constant(2., dtype=DTYPE)

	x_c = x_pos - width_mid # origin in the middle of the frame 
	y_c = y_pos - height_mid # origin in the middle of the frame

	radius = tf.sqrt(tf.pow(x_c, 2)+tf.pow(y_c, 2))
	radius = tf.expand_dims(radius, 1)

	angle = tf.atan2(y_c, x_c) # radians
	angle = angle/tf.constant(np.pi, dtype=DTYPE)*180.  # degrees
	theta = tf.math.mod(angle, 360.) # bound up to 1 circumference  
	theta = tf.expand_dims(theta, 1)
	rot_theta = theta - rot_angles # rotate angles 
	rot_theta = tf.experimental.numpy.deg2rad(rot_theta)

	x_s = tf.multiply(radius, tf.cos(rot_theta)) + width_mid
	y_s = tf.multiply(radius, tf.sin(rot_theta)) + height_mid
	return x_s, y_s

def get_window(frame, x, y, size):
    x = tf.cast(x, DTYPEINT)
    y = tf.cast(y, DTYPEINT)
    size = tf.cast(size, DTYPEINT)
    width, height = frame.shape
    width = tf.cast(width, DTYPEINT)
    height = tf.cast(height, DTYPEINT)
    start_w = tf.maximum(tf.constant(0, dtype=DTYPEINT), y - size//2)
    start_h = tf.maximum(tf.constant(0, dtype=DTYPEINT), x - size//2)
    if size % 2 != 0:
        end_w   = tf.minimum(width, y + size//2+1)
        end_h   = tf.minimum(height, x + size//2+1)
    else:
        end_w   = tf.minimum(width, y + size//2)
        end_h   = tf.minimum(height, x + size//2)

    # Cut the window from the cube
    window = tf.slice(frame, [start_w, start_h], [end_w - start_w, end_h - start_h])
    return window

def cut_patches(x, y, flux, fwhm, cube, psf, ids, size=15):
	windows = tf.map_fn(lambda x: get_window(x[0], x[1], x[2], size),
			  			(cube, x, y),
			  			fn_output_signature=DTYPE)

	psf_new_shape = tf.map_fn(lambda x: reshape_image(x, tf.shape(windows)[-1]), psf, 
						 fn_output_signature=DTYPE)

	return x, y, flux, fwhm, cube, psf_new_shape, windows, ids

def select_and_flat(x,y,flux, fwhm, cube, psf, windows, ids):
	flux = tf.tile(tf.expand_dims(flux, 0), [tf.shape(windows)[0]])

	fwhm = tf.tile(tf.expand_dims(fwhm, 0), [tf.shape(windows)[0]])
	psf = tf.tile(tf.expand_dims(psf, 0), [tf.shape(windows)[0], 1, 1, 1])
	ids = tf.tile(tf.expand_dims(ids, 0), [tf.shape(windows)[0]])

	n_frames = tf.shape(psf)[0]
	n_psfs   = tf.shape(psf)[1]
	step_slice = tf.cast(n_frames/n_psfs, DTYPEINT)
	pivot = tf.range(0, n_frames, step_slice, dtype=DTYPEINT)
	indices = tf.range(0, tf.shape(pivot)[0], dtype=DTYPEINT)

	psf = tf.cast(psf, DTYPE)
	psf_flat = tf.map_fn(lambda x: tf.slice(psf, [x[0], x[1], 0, 0], [step_slice, 1, -1, -1]),
						(pivot, indices), fn_output_signature=DTYPE)
	psf_flat = tf.reshape(psf_flat, [n_frames, tf.shape(psf)[-2], tf.shape(psf)[-1]] )

	coords = tf.stack([x, y], 1)
	return tf.data.Dataset.from_tensor_slices((windows, psf_flat, flux, cube, fwhm, ids, coords))

def augment_dataset(windows, psf, flux, cube, fwhm, ids, coords):
	windows = tf.expand_dims(windows, -1),
	psf = tf.expand_dims(psf, -1),
	rand_k = tf.random.uniform([1], minval=0, maxval=4, dtype=DTYPEINT)
	rand_k = tf.squeeze(rand_k)
	rotated_windows = tf.image.rot90(windows, k=rand_k, name=None)
	rotated_psf = tf.image.rot90(psf, k=rand_k, name=None)
	rotated_windows = tf.squeeze(rotated_windows, axis=0)
	rotated_psf = tf.squeeze(rotated_psf, axis=0)
	return rotated_windows, rotated_psf, flux, cube, fwhm, ids, coords

def create_input_dictonary(windows, psf, flux, cube, fwhm, ids, coords):
	inputs = {
		'windows': tf.expand_dims(windows, -1),
		'psf': tf.expand_dims(psf, -1),
		'flux': flux,
	}
	outputs = {
		'fwhm': fwhm,
		'cube':cube,
		'ids': ids,
		'coords':coords
	}
	return inputs, outputs


def create_dataset(cube, psf, rot_angles, table, batch_size=10, window_size=15, repeat=1):    
    numpy_table = table.values

    x_pos = numpy_table[:, 0]
    y_pos = numpy_table[:, 1]
    flux  = numpy_table[:, 2]
    fwhm  = numpy_table[:, 3]

    x_rot, y_rot = get_rotated_coords(x_pos, y_pos, cube, rot_angles)

    cube = tf.expand_dims(cube, 0)
    cube = tf.tile(cube, [numpy_table.shape[0], 1, 1, 1])

    psf = tf.expand_dims(psf, 0)
    psf = tf.tile(psf, [numpy_table.shape[0], 1, 1, 1])

    rot_angles = tf.expand_dims(rot_angles, 0)
    rot_angles = tf.tile(rot_angles, [numpy_table.shape[0], 1])

    ids = np.arange(numpy_table.shape[0])
    
    dataset = tf.data.Dataset.from_tensor_slices((x_rot, y_rot, flux, fwhm, cube, psf, ids))
    dataset = dataset.map(lambda a,b,c,d,e,f,g: cut_patches(a,b,c,d,e,f,g,size=window_size))
    dataset = dataset.flat_map(select_and_flat)
    dataset = dataset.repeat(repeat)
#     dataset = dataset.map(augment_dataset)
    dataset = dataset.map(create_input_dictonary)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(2)
    return dataset