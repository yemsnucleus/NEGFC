import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import argparse
import names
import sys
import os

from core.data import preprocess_and_save, get_companions, create_tf_dataset
from vip_hci.preproc.derotation import cube_derotate
from tensorflow.keras.optimizers import Adam
from core.model import create_model

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    WEIGHTS_FOLDER = os.path.join(opt.p, names.get_first_name())

    cube, psf, rot_angles, table = preprocess_and_save(opt.data, lambda_ch=0)

    # cut windows 
    cube = cube_derotate(cube, rot_angles, nproc=4, imlib='opencv', interpolation='nearneig')

    window_size = 30
    table = table[table['snr'] > 70.]

    optimal_fluxes = []
    for index, row in table.iterrows():
        companion = get_companions(cube, x=row['x'], y=row['y'], window_size=window_size)
        psf       = get_companions(psf, x=psf.shape[-1]//2, y=psf.shape[-1]//2, window_size=window_size)

        loader, input_shape = create_tf_dataset(psf, companion, batch_size=1, repeat=1)

        model = create_model(input_shape=input_shape, init_flux=row['flux'])

        model.compile(optimizer=Adam(1e-2))

        es  = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

        hist = model.fit(loader, epochs=100000, callbacks=[es], workers=4, use_multiprocessing=True, verbose=0)

        model.save_weights(os.path.join(WEIGHTS_FOLDER, f'model_{index}', 'weights'))

        best_epoch = np.argmin(hist.history['loss'])
        best_flux = hist.history['flux'][best_epoch]
        optimal_fluxes.append(best_flux)
    
    table['optimal_flux'] = optimal_fluxes
    table.to_csv(os.path.join(WEIGHTS_FOLDER, 'prediction.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', default='./logs/test', type=str,
                    help='Project path to save logs and weigths')
    parser.add_argument('--data', default='./data/real/dhtau', type=str,
                    help='folder containing the cube, psfs, and rotational angles')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Earlystopping threshold in number of epochs')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--ws', default=63, type=int,
                        help='windows size of the PSFs')

    opt = parser.parse_args()
    run(opt)


# N = 5
# fig, axes = plt.subplots(N, 2, sharex=True, sharey=True, dpi=300, 
#                         gridspec_kw={'hspace': 0.1, 'wspace': -0.8})
# for i, x in enumerate(dataset.take(N)):
#     axes[i][0].imshow(x['input'])
#     axes[i][1].imshow(x['output'])

# fig.savefig('./output/translated.png', bbox_inches='tight')