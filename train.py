import tensorflow as tf

from src.model import create_embedding_model, create_flux_model, create_convnet
from src.preprocess import preprocess_folder
from src.format_data import create_dataset
from src.losses import wrapper, keep_back

from tensorflow.keras.optimizers import Adam
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


cube, psf, rot_angles, table = preprocess_folder(root='./data/f_dhtau', 
                                                 target_folder='./data/f_dhtau/preprocessed')

window_size = 50
back_q = 10
table = table[table['snr'] > 10]

dataset = create_dataset(cube, 
                         psf, 
                         rot_angles, 
                         table, 
                         window_size=window_size, 
                         batch_size=16, 
                         repeat=1,
                         back_q=back_q)

model = create_convnet(window_size=config.window_size)

optimizer = Adam(config.learning_rate)
loss_fn = wrapper(keep_back, decay_factor=config.decay_factor)

model.compile(loss_fn=loss_fn, optimizer=optimizer)

es = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=1e-3,
        patience=50,
        mode='min',
        restore_best_weights=True,
    )
wandb_cb = WandbModelCheckpoint(filepath=os.path.join(WEIGHTS_FOLDER, 'model'),
                                                      monitor='loss',
                                                      save_freq='epoch',
                                                      save_weights_only=True, 
                                                      save_best_only=True)

wandb_mlog = WandbMetricsLogger(log_freq='epoch')

hist = model.fit(dataset, epochs=10000, callbacks=[es, wandb_cb, wandb_mlog])
model.save_weights(os.path.join(WEIGHTS_FOLDER, 'weigths'))