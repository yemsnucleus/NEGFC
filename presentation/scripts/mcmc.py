import matplotlib.pyplot as plt
import numpy as np
import argparse
import names
import time
import toml
import os

from vip_hci.preproc.recentering import frame_center, cube_recenter_2dfit
from vip_hci.preproc.badframes import cube_detect_badfr_correlation
from vip_hci.fm.negfc_mcmc import mcmc_negfc_sampling
from vip_hci.preproc.derotation import cube_derotate
from vip_hci.psfsub import median_sub, pca_annulus
from vip_hci.var import fit_2dgaussian
from vip_hci.fm import normalize_psf

from astropy.io import fits

def create_circular_mask(height, width, center=None, radius=None):
    if center is None:
        center = (height//2, width//2)
    if radius is None:
        radius = min(center[0], center[1], height-center[0], width-center[1])
        
    Y, X = np.meshgrid(np.arange(height), np.arange(width))
     
    dist_from_center = np.linalg.norm(np.stack([Y - center[0], X - center[1]], axis=-1), axis=-1)
    mask = np.where(dist_from_center <= radius, 0., 1.)
    return mask

def parse_filter_code(code):
    ftype = code.split('_')[0]
    filters = code.split('_')[1]
    if ftype == 'DB':
        #DUAL BAND
        filter_letter = filters[0]
        filters = [filter_letter+'_'+x for x in filters[1:]]
    return filters

def center_normalize_psf(psf, cropsize=30, nproc=1):
    fit = fit_2dgaussian(psf[0], #first PSF as reference 
                         crop=True, 
                         cropsize=cropsize, 
                         debug=False, 
                         full_output=True)  
    fwhm_sphere = np.mean([fit.fwhm_y, fit.fwhm_x])
    y_cent, x_cent = frame_center(psf)
    y_c=int(y_cent)
    x_c=int(x_cent)
    
    psf_center, y0_sh, x0_sh = cube_recenter_2dfit(psf, 
                                                   (y_c, x_c), 
                                                   fwhm_sphere,
                                                   model='gauss',
                                                   nproc=nproc, 
                                                   subi_size=7, 
                                                   negative=False,
                                                   full_output=True, 
                                                   plot=False,
                                                   debug=False,
                                                   verbose=False)
    psf_norm = normalize_psf(psf_center, 
                            fwhm=fwhm_sphere, 
                            size=None, 
                            threshold=None, 
                            mask_core=None,
                            full_output=False, 
                            verbose=False) 
    return psf_norm, fwhm_sphere

def clean_dataset(cube, angles=None, ref_frame=0, cropsize=180, threshold=0.85):
    goodframes, badframes = cube_detect_badfr_correlation(cube,
                                                         ref_frame,
                                                         crop_size=cropsize,
                                                         dist='spearman',
                                                         threshold=threshold,
                                                         plot=False)
    corr_cube = np.array(cube[goodframes], dtype='float32')
    if angles is not None:
        corr_ang = np.array(angles[goodframes], dtype='float32')
        return corr_cube, corr_ang
    return corr_cube

def run(opt):
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    ROOT = './presentation/results/mcmc'
    WEIGHTS_FOLDER = os.path.join(ROOT, opt.exp_name)
    os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

    # ==========================================================
    # Loading data =============================================
    # ==========================================================
    cube, header = fits.getdata(os.path.join(opt.data, 'center_im.fits'), header=True)
    psf  = fits.getdata(os.path.join(opt.data, 'median_unsat.fits'))
    ang  = fits.getdata(os.path.join(opt.data, 'rotnth.fits'))
    ang  = -ang
    filter_name = header['HIERARCH ESO INS COMB IFLT']
    filters = parse_filter_code(filter_name)
    
    # ==========================================================
    # Initial [radius - theta - Flux] ==========================
    # ==========================================================
    init_params = [344.35538038, 257.24436263, 5281.53524647] 

    # ==========================================================
    # Normalizing PSF ==========================================
    # ==========================================================
    norm_psf, fwhm = center_normalize_psf(psf[opt.ch], cropsize=30)

    # ==========================================================
    # Removing Bad Frames ======================================
    # ==========================================================
    cube, parallactic_angles = clean_dataset(cube[opt.ch], angles=ang)

    # ==========================================================
    # Crop CUBE to improve performance =========================
    # ==========================================================
    # radius * sin/cos(theta - angles) = x,y
    centerx = cube.shape[1]/2
    centery = cube.shape[2]/2
    # Mask center and Crop Frame
    mask = create_circular_mask(cube.shape[1], cube.shape[2], 
                                center=(centerx, centery), 
                                radius=5*fwhm)  
    cube = cube * mask
    margin = int(fwhm*3) 
    cube = cube[:, 
                int(centerx)-int(init_params[0])-margin : int(centerx)+int(init_params[0])+margin, 
                int(centery)-int(init_params[0])-margin : int(centery)+int(init_params[0])+margin]
    posx = init_params[0] * np.cos(np.deg2rad(init_params[1]) - np.deg2rad(parallactic_angles)) + cube.shape[1]/2
    posy = init_params[0] * np.sin(np.deg2rad(init_params[1]) - np.deg2rad(parallactic_angles)) + cube.shape[2]/2
    fig, axes = plt.subplots(1, 1, dpi=300)
    axes.imshow(cube[0]+cube[-1], origin='lower')
    axes.scatter(posx[0], posy[0], s=1, marker='x', color='r')
    axes.scatter(posx[-1], posy[-1], s=1, marker='x', color='r')
    fig.savefig(os.path.join(WEIGHTS_FOLDER, 'input_cube.png'))

    # ==========================================================
    # MCMC =====================================================
    # ==========================================================
    algo_params = {'algo': median_sub,
                   'annulus_width': 4*fwhm,
                   'imlib': 'opencv', # rotation library
                   'interpolation': 'lanczos4'}
    algo_options = {'fwhm': fwhm, 
                    'verbose':False, 
                    'imlib': 'opencv', # rotation library
                    'interpolation': 'lanczos4'}
    conv_params = {'conv_test': 'ac', # https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
                   'ac_c': 50,
                   'ac_count_thr': 1,
                   'check_maxgap': 50}
    mcmc_params = {'nwalkers': opt.nwalkers,
                   'niteration_min': min([100, opt.niter]),
                   'niteration_limit': opt.niter,
                   'bounds': None,
                   'sigma':'spe',
                   'nproc': opt.nproc}
    negfc_params = {'mu_sigma': True,
                    'aperture_radius': 2.}

    pixel_scale = 12.255/1000 
    chain = mcmc_negfc_sampling(cube=cube,
                        angs=parallactic_angles, 
                        psfn=norm_psf[0], # First psf 
                        initial_state=init_params,
                        ncomp=1,
                        **algo_params, 
                        **negfc_params,
                        **mcmc_params, 
                        **conv_params,
                        display=True, 
                        verbosity=0, 
                        algo_options=algo_options,
                        save=True, 
                        output_dir=WEIGHTS_FOLDER)
    end = time.time()                    
    print('TIME ELAPSED: ', end - start)

    conf_hparams = opt.__dict__
    conf_hparams['execution_time'] = np.round(end - start, 4)

    with open(os.path.join(project_path, 'config.toml'), 'w') as f:
        toml.dump(conf_hparams, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/real/eta_tel_b', type=str,
                    help='Data directory')

    parser.add_argument('--exp-name', default='', type=str,
                    help='test')

    parser.add_argument('--ch', default=0, type=float,
                        help='Channel filter ID')

    parser.add_argument('--nproc', default=1, type=int,
                        help='Number of process to run distributed')

    parser.add_argument('--nwalkers', default=100, type=int,
                        help='Number of walkers')
    parser.add_argument('--niter', default=10000, type=int,
                        help='Maximum number of iterations')


    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')


    opt = parser.parse_args()
    run(opt)