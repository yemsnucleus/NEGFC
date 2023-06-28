import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
import os

from src.record import save_records, load_records
from src.preprocess import preprocess_folder
from src.model import create_model
from src.losses import reduce_std

from tensorflow.keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


name_ds = sys.argv[2]
window_size = 50 
WEIGHTS_FOLDER = './logs/test_{}'.format(name_ds)

if not os.path.exists('./data/records/{}/fold_0/train'.format(name_ds)):
    cube, psf, rot_angles, table = preprocess_folder(
        root='./data/{}'.format(name_ds), 
        target_folder='./data/{}/preprocessed'.format(name_ds))
    

    print(table[['flux', 'true_flux', 'snr']])
    snrthreshold = input('Enter a SNR threshoold to filter the table: ')
    
    
    save_records(cube, psf, rot_angles, table, 
                 output_path='./data/records/{}'.format(name_ds), 
                 snr_threshold=float(snrthreshold),
                 window_size=50,
                 train_val_test=(0.5, 0.2, 0.3))


train_ds = load_records('./data/records/{}/fold_0/train'.format(name_ds), 
                        batch_size=128, 
                        repeat=100,
                        augmentation=True)
val_ds = load_records('./data/records/{}/fold_0/val'.format(name_ds),
                      batch_size=128)

model = create_model(window_size=window_size)

# for index, x in enumerate(train_ds):

#     fig, axes = plt.subplots(1, 4, dpi=300)
#     axes[0].imshow(x['cube'][0][0])
#     axes[1].imshow(x['cube'][0][1])
#     axes[2].imshow(x['psf'][0][0])
#     axes[3].imshow(x['psf'][0][1])
#     plt.savefig('./output/rot_{}.png'.format(index), bbox_inches='tight')

#     y_pred = model(x)
#     loss = reduce_std(x, y_pred)
#     print(loss)

optimizer = Adam(1e-3)
model.compile(loss_fn=reduce_std, optimizer=optimizer)

es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=50,
        mode='min',
        restore_best_weights=True,
    )
tb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(WEIGHTS_FOLDER, 'logs'),
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch')

hist = model.fit(train_ds, epochs=10000, validation_data=val_ds, callbacks=[es, tb])
model.save_weights(os.path.join(WEIGHTS_FOLDER, 'weigths'))
