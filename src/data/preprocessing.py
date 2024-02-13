import numpy as np


from vip_hci.preproc.recentering import frame_shift, frame_center
from vip_hci.preproc.cosmetics import cube_crop_frames
from joblib	import Parallel, delayed


def parse_filter_code(code):
    """ Read the code and write the filters separated.
    
    Args:
        code (str): Code string from X
    
    Returns:
        TYPE: Description
    """
    ftype = code.split('_')[0]
    filters = code.split('_')[1]
    if ftype == 'DB':
        #DUAL BAND
        filter_letter = filters[0]
        filters = [filter_letter+'_'+x for x in filters[1:]]
    return filters

def modify_shape_and_center(img, shift_h=1, shift_w=1):
    # Get the height and width of the image
    height, width = img.shape[:2]

    # Increase the image size by 1 pixel
    new_height = height + shift_h
    new_width = width + shift_w

    # Create a new image with the increased size
    new_img = np.zeros((new_height, new_width))

    # Calculate the offset needed to center the original image in the new image
    x_offset = int((new_width - width) / 2)
    y_offset = int((new_height - height) / 2)

    # Copy the original image into the center of the new image
    new_img[y_offset:y_offset+height, x_offset:x_offset+width] = img

    return new_img


def shift_and_crop_cube(cube, n_jobs=1, shift_x=-1, shift_y=-1):
	"""Shift and crop each frame within a given cube

	Since VIP only works on odd frames we should rescale if necessary.

	:param cube: Sequence of frames forming the cube
	:type cube: numpy.ndarray
	:param n_jobs: Number of cores to distribute, defaults to 1
	:type n_jobs: number, optional
	:param shift_x: Number of pixels to shift in the x-axis, defaults to -1
	:type shift_x: number, optional
	:param shift_y: Number of pixels to shift in the y-axis, defaults to -1
	:type shift_y: number, optional
	:returns: A recentered and cropped cube containing even-dim frames
	:rtype: {numpy.ndarray}
	"""
	shifted_cube = Parallel(n_jobs=n_jobs)(delayed(frame_shift)(frame, 
	    														shift_x, 
													   			shift_y) \
										   for frame in cube)
	shifted_cube = np.array(shifted_cube)

	y_center, x_center = frame_center(shifted_cube[0])
	ycen   = y_center-0.5
	xcen   = x_center-0.5
	newdim = shifted_cube.shape[-1]-1

	shifted_cube = cube_crop_frames(shifted_cube,
	                                newdim,
	                                xy=[int(ycen), int(xcen)], 
	                                force=True) 
	return shifted_cube

def crop_cube(cube, size=256):
    center_x = cube.shape[-1]//2
    center_y = cube.shape[-2]//2

    offset = size//2
    return cube[..., center_y-offset:center_y+offset, center_x-offset:center_x+offset]

def radial_to_xy(cube, rad_distance, theta):
    cube_dim = cube[0].shape
    center_cube_x = cube_dim[-2]/2
    center_cube_y = cube_dim[-1]/2
    posy = rad_distance * np.sin(np.deg2rad(theta)) + center_cube_y
    posx = rad_distance * np.cos(np.deg2rad(theta)) + center_cube_x
    return posx, posy