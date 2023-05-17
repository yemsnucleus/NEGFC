import multiprocessing as mp
import pandas as pd
import numpy as np 
import time
import sys
import os

from vip_hci.fm.negfc_mcmc import mcmc_negfc_sampling
from multiprocessing import cpu_count
from vip_hci.preproc.recentering import (frame_shift, 
                                         frame_center, 
                                         cube_recenter_2dfit)
from vip_hci.var import fit_2dgaussian
from vip_hci.fm import normalize_psf
from vip_hci.fits import open_fits
from vip_hci.metrics import detection
from vip_hci.fm.negfc_simplex import firstguess
from vip_hci.psfsub import pca, pca_annulus

def get_data(root='./data/HCI', lambda_ch=0, psf_pos=0):
    cube_route = os.path.join(root, 'center_im.fits')
    cube       = open_fits(cube_route, header=False) 

    psf_route = os.path.join(root, 'median_unsat.fits')
    psf       = open_fits(psf_route, header=False)

    ra_route = os.path.join(root, 'rotnth.fits')
    rot_ang  = open_fits(ra_route, header=False)
    rot_ang  = -rot_ang

    return {'cube': cube, 
            'psf':psf,
            'rot_angles': rot_ang}

def preprocess_psf(cube):
    if len(cube[..., 0])%2 == 0:
        ycen=ycube_center-0.5
        xcen=xcube_center-0.5
        newdim=len(cube[1,:,1])-1

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

def run(root='./data/DHTau', backlog_name='backlog'):
    os.makedirs('./results/cpu/', exist_ok=True)
    df_backlog = pd.DataFrame(columns= ['step', 'time', 'y', 'x', 'flux', 
                                        'std_x', 'std_y', 'std_flux', 
                                        'med_x', 'med_y', 'med_flux'])

    backlog = []
    lambda_ch = 0
    psf_pos = 0

    # ===========================================
    # ============ NORMALIZATION ================
    # =========================================== 
    t0 = time.time()
    
    data = get_data(root)
    if data['psf'].shape[-1] % 2 == 0:
        psf_even = []
        for psf_lambda in data['psf']:
            partial = []
            for curr_psf in psf_lambda:
                partial.append(modify_shape_and_center(curr_psf, shift_h=1, shift_w=1)) 
            psf_even.append(partial)
        psf = np.array(psf_even)        
    
    single_psf = psf[lambda_ch, psf_pos, :-1, :-1]
    ceny, cenx = frame_center(single_psf)
    imside = single_psf.shape[0]
    cropsize = 30
    fit = fit_2dgaussian(single_psf, 
                         crop=True, 
                         cropsize=30, 
                         debug=False, 
                         full_output=True) 
    fwhm_sphere = np.mean([fit.fwhm_y,fit.fwhm_x]) 

    y_cent, x_cent = frame_center(single_psf) 
    y_c=int(y_cent)
    x_c=int(x_cent)
    psf_center = cube_recenter_2dfit(psf[lambda_ch], 
                                     (y_c, x_c),
                                     fwhm_sphere,
                                     model='gauss',
                                     nproc=8, 
                                     subi_size=7,
                                     negative=False,
                                     full_output=False, 
                                     debug=False)
    
    psf_norm, fwhm_flux, fwhm = normalize_psf(psf_center, 
                                         fwhm=fwhm_sphere, 
                                         size=None, 
                                         threshold=None, 
                                         mask_core=None,
                                         full_output=True, 
                                         verbose=True) 
    t1 = time.time()
    df_backlog.loc[len(df_backlog.index)] = ['psf_norm', t1-t0, '', '', '', '', '', '', '', '', '']
    df_backlog.to_csv('./results/vip/{}.csv'.format(backlog_name), index=False)

    # ===========================================
    # ====================== PCA ================
    # =========================================== 
    t0 = time.time()
    fr_pca = pca(data['cube'][lambda_ch], 
                 data['rot_angles'],
                 svd_mode='eigencupy', 
                 full_output=False,
                 imlib='opencv')
    t1 = time.time()
    df_backlog.loc[len(df_backlog.index)] = ['pca+adi', t1-t0, '', '', '', '', '', '', '', '', '']
    df_backlog.to_csv('./results/vip/{}.csv'.format(backlog_name), index=False)

    # ===========================================
    # ================ DETECTION ================
    # =========================================== 
    t0 = time.time()
    res = detection(fr_pca, 
                    fwhm=fwhm[psf_pos], 
                    psf=psf_norm[psf_pos], 
                    bkg_sigma=5, 
                    snr_thresh=2, 
                    debug=False, 
                    plot=False, 
                    verbose=True, 
                    full_output=True)
    t1 = time.time()
    df_backlog.loc[len(df_backlog.index)] = ['detection', 
                                             t1-t0, 
                                             res.iloc[0]['y'], 
                                             res.iloc[0]['x'], 
                                             '', '', '', '', '', '', '']
    df_backlog.to_csv('./results/vip/{}.csv'.format(backlog_name), index=False)

    # ===========================================
    # =============== FIRST GUESS ===============
    # =========================================== 
    t0 = time.time()
    results = firstguess(data['cube'][lambda_ch], 
                         angs=data['rot_angles'],       
                         psfn=psf_norm[psf_pos], 
                         ncomp=1, 
                         planets_xy_coord=[(res.iloc[0]['y'], res.iloc[0]['x'])],
                         imlib='opencv',
                         fwhm=fwhm[psf_pos],
                         simplex=True,
                         annulus_width=2*fwhm[psf_pos],
                         aperture_radius=2,
                         algo_options={
                             'nproc': mp.cpu_count()//2,
                             'imlib': 'opencv'
                         })    
    radius_fguess = results[0][0]
    theta_fguess  = results[1][0]
    flux_fguess   = results[2][0]
        
    centy_fr, centx_fr = frame_center(data['cube'][lambda_ch])
    posy = radius_fguess * np.sin(np.deg2rad(theta_fguess)) + centy_fr
    posx = radius_fguess * np.cos(np.deg2rad(theta_fguess)) + centx_fr

    t1 = time.time()
    df_backlog.loc[len(df_backlog.index)] = ['fguess', 
                                             t1-t0, 
                                             posy, 
                                             posx, 
                                             flux_fguess, '', '', '', '', '', '']
    df_backlog.to_csv('./results/vip/{}.csv'.format(backlog_name), index=False)

    # ===========================================
    # ================== MCMC ===================
    # =========================================== 
    t0 = time.time()    
    initial_state = [radius_fguess, theta_fguess, flux_fguess]
    chain = mcmc_negfc_sampling(data['cube'][lambda_ch], 
                                data['rot_angles'],  
                                psf_norm[psf_pos], 
                                ncomp=1, 
                                plsc=1.2, 
                                initial_state=initial_state,
                                display=True, verbosity=1, 
                                save=False, output_dir='./')    
    xm = np.mean(chain, 0)
    radius_star = np.mean(xm[:, 0])
    theta_star  = np.mean(xm[:, 1])
    flux_star   = np.mean(xm[:, 2])
    
    radius_std = np.std(xm[:, 0])
    theta_std  = np.std(xm[:, 1])
    flux_std   = np.std(xm[:, 2])

    radius_med = np.median(xm[:, 0])
    theta_med  = np.median(xm[:, 1])
    flux_med   = np.median(xm[:, 2])

    centy_fr, centx_fr = frame_center(cube[0])
    posy_star = radius_star * np.sin(np.deg2rad(theta_star)) + centy_fr
    posx_star = radius_star * np.cos(np.deg2rad(theta_star)) + centx_fr

    t1 = time.time()    
    df_backlog.loc[len(df_backlog.index)] = ['mcmc', 
                                             t1-t0, 
                                             posy_star, 
                                             posx_star, 
                                             flux_star,
                                             radius_std,
                                             theta_std,
                                             flux_std,
                                             radius_med,
                                             theta_med,
                                             flux_med]

    df_backlog.to_csv('./results/vip/{}.csv'.format(backlog_name), index=False)
    

if __name__ == '__main__':

    root_data = sys.argv[1]
    backlog_name = sys.argv[2]

    print(':'*10)
    print('[INFO] Script started!')
    print('[INFO] Loading {}'.format(root_data))
    print('[INFO] Backlogs stored at /results/cpu/{}.csv'.format(backlog_name))
    print(':'*10)
    run(root_data, backlog_name)
















