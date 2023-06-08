import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import preprocess_folder
from src.format_data import create_dataset
from src.model import create_embedding_model, create_flux_model
from src.losses import get_companion_std

from tensorflow.keras.optimizers import Adam
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cube, psf, rot_angles, table = preprocess_folder(root='./data/fake', 
												 target_folder='./data/fake/preprocessed')

window_size = 20
dataset = create_dataset(cube, 
                         psf, 
                         rot_angles, 
                         table, 
                         window_size=window_size, 
                         batch_size=2000, 
                         repeat=10)

# N = 3
# fig, axes = plt.subplots(N, 2)
# for i, (x, y) in enumerate(dataset.unbatch().take(N)):
#     axes[i][0].imshow(x['psf'] - x['windows'])
#     axes[i][1].imshow(x['windows'])
# plt.show()

model = create_flux_model(window_size=window_size)

# print(model.summary())
optimizer = Adam(1e-3)
model.compile(loss_fn=get_companion_std, optimizer=optimizer)

es = tf.keras.callbacks.EarlyStopping(
         monitor='loss',
         min_delta=1e-4,
         patience=50,
         mode='minimize',
         restore_best_weights=True,
     )

model.fit(dataset, epochs=1000, callbacks=[es])


# print(model.summary())

# winsize = 15
# for x, y in dataset:
# 	out = model(x)
# 	loss = get_companion_std(x['windows'], out)
# 	print(loss)
# 	fig, axes = plt.subplots(1, 2, dpi=300, sharey=True, sharex=True)
# 	axes[0].imshow(x['psf'][0])
# 	axes[1].imshow(out[0])
# 	plt.show()


