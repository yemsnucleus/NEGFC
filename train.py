import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import argparse
import names
import sys
import os

from src.record import load_records
from src.model import create_model
from src.losses import coords_rmse

from tensorflow.keras.optimizers import Adam


def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    WEIGHTS_FOLDER = os.path.join(opt.p, names.get_first_name())

    train_ds = load_records('{}/train.record'.format(opt.data), batch_size=opt.bs, augmentation=True)
    val_ds   = load_records('{}/val.record'.format(opt.data), batch_size=opt.bs, augmentation=True)
    
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()

    maxi = 0
    for index, (x, y) in enumerate(train_ds.take(9)):
        axes[index].set_title(str(y[index].numpy()))
        im = axes[index].imshow(x[index], vmin=0, vmax=0.1)
        axes[index].set_xticks([])
        axes[index].set_yticks([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig('./output/inputs.png', format='png')

    # print(WEIGHTS_FOLDER)
    model = create_model(opt.ws)

    optimizer = Adam(opt.lr)
    model.compile(loss_fn=coords_rmse, optimizer=optimizer)

    es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=opt.patience,
            mode='min',
            restore_best_weights=True,
        )
    tb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(WEIGHTS_FOLDER, 'logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch')

    hist = model.fit(train_ds, epochs=opt.epochs, validation_data=val_ds, callbacks=[es, tb])
    model.save_weights(os.path.join(WEIGHTS_FOLDER, 'weigths'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', default='./logs/test', type=str,
                    help='Project path to save logs and weigths')
    parser.add_argument('--data', default='./data/records/coords/fold_0', type=str,
                    help='Datasets where .records files are located')
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