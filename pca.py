import numpy as np

from vip_hci.var.shapes			import get_square, prepare_matrix
from vip_hci.preproc.parangles	import check_pa_vector
from vip_hci.preproc.derotation import cube_derotate

from plottools 					import plot_to_compare


def reduce_pca(cube, rot_angles, ncomp=1, fwhm=4, plot=False, n_jobs=1):
	""" Reduce cube using Angular Differential Imaging (ADI) techinique. 
	
	This function reduce the frame-axis dimension of the cube to 1
	We subtract the principal components to each original frame (residuals).
	Using the rotation angles we center all the frames and reduce them to the median. 
	If more than one channels were provided, we calculate the mean of the medians per wavelength.
	:param cube: A cube containing frames
	:type cube: numpy.ndarray
	:param rot_angles: Rotation angles used to center residual frames
	:type rot_angles: numpy.ndarray
	:param ncomp: Number of component to reduce in the frames axis, defaults to 1
	:type ncomp: number, optional
	:param fwhm_sphere: Full-Width at Half Maximum value to initialice the gaussian model, defaults to 4
	:type fwhm_sphere: number, optional
	:param plot: If true, displays original frame vs reconstruction, defaults to False
	:type plot: bool, optional
	:returns: Median of centered residuals
	:rtype: {np.ndarray}
	"""
	nz, ny, nx = cube.shape
	rot_angles = check_pa_vector(rot_angles)

	# Build the matrix for the SVD/PCA and other matrix decompositions. (flatten the cube)
	matrix = prepare_matrix(cube, mode='fullfr', verbose=False)

	# DO SVD on the matrix values
	U, S, V = np.linalg.svd(matrix.T, full_matrices=False)
	# `matrix.T` has dimension LxD where L is `time steps` and D is `pixels space`
	# We want to see the max. var or info in the pixels space along the time steps axis
	# Then when applying SVD,
	# The columns of U represent the principal components in time
	# The columns of V represent the principal components in feature space.
	U = U[:, :ncomp].T

	transformed   = np.dot(U, matrix.T)
	reconstructed = np.dot(transformed.T, U)
	residuals     = matrix - reconstructed

	residuals 	  = residuals.reshape(residuals.shape[0], ny, nx)
	reconstructed = reconstructed.reshape(reconstructed.shape[0], ny, nx)
	matrix 		  = matrix.reshape(matrix.shape[0], ny, nx)
	p_components  = U.reshape(U.shape[0], ny, nx)

	# NOT SURE WHY rot_angles IS NEGATIVE
	array_der = cube_derotate(residuals, -rot_angles, nproc=n_jobs, 
							  imlib='vip-fft', interpolation='lanczos4')
	res_frame = np.nanmedian(array_der, axis=0)
	if plot:
		plot_to_compare([matrix[0], reconstructed[0], residuals[0], res_frame], 
						['Original', 'Reconstructed', 'Residuals', 'median'])

	return res_frame