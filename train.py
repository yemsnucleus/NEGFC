import tensorflow as tf
import matplotlib.pyplot as plt 
import sys
import os

from src.record import save_records, load_records
from src.preprocess import preprocess_folder
from src.model import create_model
from src.losses import reduce_std

from tensorflow.keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

window_size = 50 
WEIGHTS_FOLDER = './logs/test'
# cube, psf, rot_angles, table = preprocess_folder(root='./data/f_dhtau', 
#                                                  target_folder='./data/f_dhtau/preprocessed')

# save_records(cube, psf, rot_angles, table, 
#              output_path='./data/records/f_dhtau', 
#              window_size=50,
#              train_val_test=(0.5, 0.2, 0.3))

train_ds = load_records('./data/records/f_dhtau/fold_0/train', batch_size=10, repeat=10)
val_ds = load_records('./data/records/f_dhtau/fold_0/val')

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

hist = model.fit(train_ds, epochs=10000, validation_data=val_ds, callbacks=[es])
model.save_weights(os.path.join(WEIGHTS_FOLDER, 'weigths'))