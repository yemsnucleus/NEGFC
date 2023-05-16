import tensorflow as tf 
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import time
import os

from gpu.mcmc import run_chain, run_chain_only_flux
from tensorflow.keras.optimizers import Adam 
from gpu.negfc_models import get_model
from gpu.losses import custom_loss
from vip_hci.fits import open_fits
from gpu.data import get_dataset
import gpu.fake_comp as tfnegfc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def wrapper(fn, fwhm, std=True):
    def inner(*args):
        out = fn(*args, fwhm=fwhm, std=std)
        return out
    return inner

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

def run():
    
    backlog = []
    lambda_ch = 0
    psf_pos = 0
    
    # ========= NORMALIZE =========
    t0 = time.time()
    root = './data/HCI'
    data = get_data(root)
    results = tfnegfc.adjust_gaussian(data['psf'][lambda_ch, psf_pos])
    fwhm_sphere  = tf.reduce_mean(results['fwhm'])
    centered_psf = tfnegfc.center_cube(data['psf'][lambda_ch], fwhm_sphere)
    normalized_psf = tfnegfc.normalize_psf(centered_psf, fwhm=fwhm_sphere)
    t1 = time.time()
    backlog.append(['psf norm', t1-t0, '', '', ''])
    
    # =========== PCA + ADI =========
    t0 = time.time()
    adi_image = tfnegfc.apply_adi(data['cube'][lambda_ch], 
                                  data['rot_angles'], 
                                  out_size=data['cube'][lambda_ch].shape, 
                                  ncomp=1, 
                                  derotate='tf')
    t1 = time.time()
    backlog.append(['pca+adi', t1-t0, '', '', ''])
    
    # ============ DETECTION =========
    t0 = time.time()
    table = tfnegfc.get_coords(adi_image.numpy(), 
                               fwhm=fwhm_sphere, 
                               bkg_sigma=5, 
                               cut_size=10)
    t1 = time.time()
    backlog.append(['detection', 
                t1-t0, 
                table.iloc[0]['y'], 
                table.iloc[0]['x'], 
                table.iloc[0]['flux']])
    
    # ======= OPTIMIZATION ==========
    
    nfwhm = 4
    custom_loss_w = wrapper(custom_loss, 
                            fwhm=table.iloc[0]['fwhm_mean']*nfwhm, 
                            std=False)

    dataset, recovery = get_dataset(data['cube'], 
                                    normalized_psf, 
                                    data['rot_angles'], 
                                    normalize=0)

    model = get_model(x_init=table.iloc[0]['x'], 
                  y_init=table.iloc[0]['y'], 
                  cube=data['cube'])    

    model.compile(loss_fn=custom_loss_w, optimizer=Adam(5))
    
    
    es = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=30,
    )

    hist = model.fit(dataset, epochs=10000, verbose=1, callbacks=[es])
    
    
    x_firstguess = model.trainable_variables[0]
    y_firstguess = model.trainable_variables[1]
    flux_firstguess = model.trainable_variables[2]
    flux_firstguess = flux_firstguess[0]
    t1 = time.time()
    backlog.append(['simplex', 
                t1-t0, 
                y_firstguess, 
                x_firstguess, 
                flux_firstguess])
    
    # ======= MCMC ==========
    init_state = [flux_firstguess]
    results = run_chain_only_flux(init_state, 
                                  x_firstguess, 
                                  y_firstguess, 
                                  table.iloc[0]['fwhm_mean']*nfwhm//2, 
                                  data['cube'][lambda_ch], 
                                  normalized_psf[0], 
                                  data['rot_angles'], 
                                  num_results=10000)
    t1 = time.time()
    opt_values = [np.mean(samples_chain) for samples_chain in samples]

    backlog.append(['mcmc', 
                    t1-t0, 
                    opt_values[1], 
                    opt_values[0], 
                    opt_values[2]])
    
    df = pd.DataFrame(np.array(backlog), 
                      columns= ['step', 'time', 'y', 'x', 'flux'])
    df.to_csv('./results/tf/perform.csv')
    
    
run()
    
    
    
    
    
    
    