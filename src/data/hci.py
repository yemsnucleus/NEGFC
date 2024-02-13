import vip_hci as vip
import numpy as np
from vip_hci.preproc.recentering import frame_center, cube_recenter_2dfit
from vip_hci.fm.negfc_simplex import firstguess
from vip_hci.psfsub import pca

def normalize_psf(psf):
    psf_list = []
    fwhm = []
    for lambda_psf in psf:
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

        psf_list.append(psf_center)
        fwhm.append(fwhm_sphere)

    psf_list = np.array(psf_list)
    fwhm = np.array(fwhm)
    return psf_list, fwhm


def perform_pca(cube, q_angles, **kwargs):
    cube_list = []
    for curr_cube in cube:
        fr_pca = pca(curr_cube, 
                     q_angles,
                     full_output=False,
                     **kwargs)
        cube_list.append(fr_pca)
    return np.array(cube_list)


def detection(frames, fwhm, norm_psf):

    responses = []
    for index, curr_frame in enumerate(frames):
        res = vip.metrics.detection(curr_frame, 
                                    fwhm=fwhm[index], 
                                    psf=norm_psf[index, 0], 
                                    bkg_sigma=5, 
                                    snr_thresh=5, 
                                    debug=False, 
                                    plot=False, 
                                    verbose=False, 
                                    full_output=True)
        responses.append(res)

    return responses


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
                         annulus_width=4*fwhm,
                         aperture_radius=2,
                         algo_options={
                             'nproc': 32,
                             'imlib': 'opencv'
                         })
    return results
