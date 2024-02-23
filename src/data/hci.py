import vip_hci as vip
import numpy as np
from vip_hci.preproc.recentering import frame_center, cube_recenter_2dfit
from vip_hci.fm.negfc_simplex import firstguess
from vip_hci.psfsub import pca
from vip_hci.fm                     import normalize_psf as normalize_psf_vip
from astropy.stats                  import sigma_clipped_stats, gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling               import models, fitting
from skimage.feature                import peak_local_max
from vip_hci.var.shapes             import get_square
from vip_hci.metrics.snr_source     import snr
import pandas as pd
import time

def normalize_psf(psf):
    psf_list = []
    fwhm = []
    times = []
    for lambda_psf in psf:
        start_time = time.time()
        curr_psf = lambda_psf[0]
        ceny, cenx = frame_center(curr_psf)
        imside = curr_psf.shape[0]
        cropsize = 30

        fit = vip.var.fit_2dgaussian(curr_psf, 
                                     crop=True, 
                                     cropsize=30, 
                                     debug=False, 
                                     full_output=True) 


        fwhm_sphere = np.mean([fit.fwhm_y,fit.fwhm_x]) 


        y_cent, x_cent = frame_center(curr_psf) 
        y_c=int(y_cent)
        x_c=int(x_cent)
        psf_center, y0_sh, x0_sh = cube_recenter_2dfit(lambda_psf, (y_c, x_c), 
                                                       fwhm_sphere,
                                                       model='gauss',
                                                       nproc=8, 
                                                       subi_size=7, 
                                                       negative=False,
                                                       full_output=True, 
                                                       plot=False,
                                                       debug=False)
    
        psf_norm, fwhm_flux, fwhm_val = normalize_psf_vip(psf_center, 
                                                  fwhm=fwhm_sphere,
                                                  full_output=True, 
                                                  verbose=False) 

        psf_list.append(psf_norm)
        fwhm.append(np.mean(fwhm_val))
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    psf_list = np.array(psf_list)
    fwhm = np.array(fwhm)
    
    return psf_list, fwhm, np.array(times)


def perform_pca(cube, q_angles, **kwargs):
    cube_list = []
    times = []
    for curr_cube in cube:
        start_time = time.time()
        fr_pca = pca(curr_cube, 
                     q_angles,
                     full_output=False,
                     **kwargs)
        cube_list.append(fr_pca)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
    return np.array(cube_list), np.array(times)


def detection(frames, fwhm, norm_psf):

    responses = []
    times = []
    for index, curr_frame in enumerate(frames):
        start_time = time.time()
        res = get_intersting_coords(curr_frame, 
                                    norm_psf[index, 0], 
                                    fwhm=fwhm[index], 
                                    bkg_sigma = 5, 
                                    plot=False)
        responses.append(res)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    return responses, np.array(times)

def get_intersting_coords(frame, psf_norm, fwhm=4, bkg_sigma = 5, plot=False):
    """Get coordinates of potential companions
    
    This method infer the background noise and find coordinates that contains most luminous 
    sources. After getting coordinates, it filter them by:
        1. checking that the amplitude is positive > 0
        2. checking whether the x and y centroids of the 2d gaussian fit
           coincide with the center of the subimage (within 2px error)
        3. checking whether the mean of the fwhm in y and x of the fit are close to 
           the FWHM_PSF with a margin of 3px
    :param frame: Reduced 2-dim image after applying reduce_pca
    :type frame: np.ndarray
    :param fwhm: full-width at half maximum comming from the normalized PSF, defaults to 4
    :type fwhm: number, optional
    :param bkg_sigma: The number of standard deviations to use for both the lower and upper clipping limit, defaults to 5
    :type bkg_sigma: number, optional
    :param plot: If true, displays original frame vs reconstruction, defaults to False
    :type plot: bool, optional
    :returns: A set of coordinates of potential companions and their associated fluxes
    :rtype: {List of pairs, List of pairs}
    """
    #Calculate sigma-clipped statistics on the provided data.
    _, median, stddev = sigma_clipped_stats(frame, sigma=bkg_sigma, maxiters=None)
    bkg_level = median + (stddev * bkg_sigma)

    # Padding the image with zeros to avoid errors at the edges
    pad_value = 10
    array_padded = np.pad(frame, pad_width=pad_value, mode='constant', constant_values=0)
    
    # plot_to_compare([frame, array_padded], ['Original', 'Padded'])

    # returns the coordinates of local peaks (maxima) in an image.
    coords_temp = peak_local_max(frame, threshold_abs=bkg_level,
                                 min_distance=int(np.ceil(fwhm)),
                                 num_peaks=20)

    # CHECK BLOBS =============================================================
    y_temp = coords_temp[:, 0]
    x_temp = coords_temp[:, 1]
    coords, fluxes, fwhm_mean = [], [], []

    # Fitting a 2d gaussian to each local maxima position
    for y, x in zip(y_temp, x_temp):
        subsi = 3 * int(np.ceil(fwhm)) # Zone to fit the gaussian
        if subsi % 2 == 0:
            subsi += 1

        scy = y + pad_value
        scx = x + pad_value

        subim, suby, subx = get_square(array_padded, 
                                       subsi, scy, scx,
                                       position=True, force=True,
                                       verbose=False)
        cy, cx = frame_center(subim)
        gauss = models.Gaussian2D(amplitude=subim.max(), x_mean=cx,
                                  y_mean=cy, theta=0,
                                  x_stddev=fwhm*gaussian_fwhm_to_sigma,
                                  y_stddev=fwhm*gaussian_fwhm_to_sigma)

        sy, sx = np.indices(subim.shape)
        fitter = fitting.LevMarLSQFitter()
        fit = fitter(gauss, sx, sy, subim)
        
        if plot:
            y, x = np.indices(subim.shape)  
            plot_to_compare([subim.T, fit(x, y).T], ['subimage', 'gaussian'], dpi=100, 
                text_box='Companion candidate and its adjusted gaussian model. \
                Here we find an approximated set of coordinates and flux associated to the companion.')

        fwhm_y = fit.y_stddev.value * gaussian_sigma_to_fwhm
        fwhm_x = fit.x_stddev.value * gaussian_sigma_to_fwhm
        mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)])
        # Filtering Process
        condyf = np.allclose(fit.y_mean.value, cy, atol=2)
        condxf = np.allclose(fit.x_mean.value, cx, atol=2)
        condmf = np.allclose(mean_fwhm_fit, fwhm, atol=3)
        if fit.amplitude.value > 0 and condxf and condyf and condmf:
            coords.append((suby + fit.y_mean.value,
                           subx + fit.x_mean.value))
            fluxes.append(fit.amplitude.value)
            fwhm_mean.append(mean_fwhm_fit)

    coords = np.array(coords)
    yy = coords[:, 0] - pad_value
    xx = coords[:, 1] - pad_value
    
    table = pd.DataFrame()
    table['x']    = xx
    table['y']    = yy
    table['flux'] = fluxes
    table['fwhm_mean'] = fwhm_mean
    try:
        table['snr']  = table.apply(lambda col: snr(frame, 
                                                    (col['x'], col['y']), 
                                                    fwhm, False, verbose=False), axis=1)
    except:
        table['snr'] = -1

    return table

def optimize(cube, q_angles, norm_psf, fwhm, response, simplex=False):
    xycords = []
    for i, row in response.iterrows():
        xycords.append((row['x'], row['y']))

    results = firstguess(cube, 
                         angs=q_angles,       
                         psfn=norm_psf, 
                         ncomp=2, 
                         planets_xy_coord=xycords,
                         imlib='opencv',
                         fwhm=fwhm,
                         simplex=simplex,
                         annulus_width=2*fwhm,
                         aperture_radius=2,
                         algo_options={
                             'nproc': 32,
                             'imlib': 'opencv'
                         })
    return results
