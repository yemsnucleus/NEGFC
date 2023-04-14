import numpy as np
import vip_hci as vip
from vip_hci.var.shapes			import get_square, prepare_matrix
from vip_hci.preproc.parangles	import check_pa_vector
from vip_hci.preproc.derotation import cube_derotate

from plottools 					import plot_to_compare

from sklearn.decomposition import PCA

def reduce_pca_opt(cube, rot_angles, ncomp=1, fwhm=4, plot=False, return_cube=False, dpi=100, text_box='', n_jobs=1):

	nz, ny, nx = cube.shape
	rot_angles = check_pa_vector(rot_angles)

	pca_model = PCA(n_components=ncomp)
	matrix = prepare_matrix(cube, mode='fullfr', verbose=False)
	print(matrix.shape)
	matrix_reduced = pca_model.fit_transform(matrix)
	print(matrix_reduced.shape)
	matrix_reconstructed = pca_model.inverse_transform(matrix_reduced)
	print(matrix_reconstructed.shape)

	matrix_reconstructed = matrix_reconstructed.reshape(matrix_reconstructed.shape[0], ny, nx)

	residuals     = cube - matrix_reconstructed

	# NOT SURE WHY rot_angles IS NEGATIVE
	array_der = cube_derotate(residuals, rot_angles, nproc=n_jobs, 
							  imlib='opencv', interpolation='nearneig')

	res_frame = np.nanmedian(array_der, axis=0)

	return res_frame, residuals

def reduce_pca(cube, rot_angles, ncomp=1, fwhm=4, plot=False, return_cube=False, dpi=100, text_box='', n_jobs=1):
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
	:param return_cube: If return residual cube without collapsing frames, default to False
	:type return_cube: bool, optional
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
	array_der = cube_derotate(residuals, rot_angles, nproc=n_jobs, 
							  imlib='opencv', interpolation='nearneig')

	# plot_to_compare([residuals[0], array_der[0]], ['Original', 'Rotated'])

	res_frame = np.nanmedian(array_der, axis=0)
	if plot:
		plot_to_compare([matrix[0], reconstructed[0], array_der[0], res_frame], 
						['Original', 'Reconstructed', 'Residuals', 'Median'], 
						dpi=dpi, text_box=text_box)
	#vip.fits.write_fits('./figures/fr_pca.fits', res_frame)
	if return_cube:
		return res_frame, residuals
	return res_frame
