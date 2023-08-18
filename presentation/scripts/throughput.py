import pandas as pd
import numpy as np
import pickle
import os
import time

from photutils.aperture import aperture_photometry, CircularAperture
from core.metrics import inject_companion, get_rings, get_throughput, get_aperture_photometry, get_contrast
from core.data import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

datapath ='./data/real/pedro/'
predlogs = './logs/pedro/'

cube, psfs, rot_angles = load_data(datapath)

params_table = pd.read_csv(os.path.join(predlogs, 'prediction.csv'))

row = params_table.iloc[0]
pixel = 0.01225
num_rings =  40
cube = cube[0] # single wavelength
psfs = psfs[0] # single wavelength

cube_center = cube.shape[-1]/2

radius = np.sqrt((-cube_center)**2+(row['optimal_y']-cube_center)**2)

injected = inject_companion(row['optimal_x'], 
                            row['optimal_y'], 
                            row['optimal_flux'],
                            cube=np.zeros_like(cube),
                            psf=psfs[0],
                            rot_angles=rot_angles)

print('[INFO] Getting {} rings'.format(num_rings))
clean_cube = cube - injected #
regions, rad_dist_px = get_rings(row['optimal_x'], row['optimal_y'], 
                                 fhwm=row['fwhm_mean'],
                                 cube=clean_cube, 
                                 rot_angles=rot_angles,
                                 num_rings=num_rings)

print('[INFO] Calculating Throughput. You may take the day')
start = time.time()
throughput, opt_summary = get_throughput(
                            clean_cube, 
                            psfs, 
                            rot_angles,
                            fwhm=row['fwhm_mean'],
                            rad_distances=rad_dist_px,
                            regions=regions,
                            K=4,
                            window_size=10,
                            optimize=True)
end = time.time()
print('ELAPSED: ', end - start)

opt_summary.to_csv(os.path.join(predlogs, 'opt_summary.csv'), index=False)
with open(os.path.join(predlogs, 'throughput.pkl'), "wb") as fp:   #Pickling
    pickle.dump(throughput, fp)
    
    
print('ELAPSED: ', end - start)