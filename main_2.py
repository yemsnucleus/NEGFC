import matplotlib.pyplot as plt

from src.preprocess import preprocess_folder
from src.format_data import create_dataset
from src.model import create_embedding_model
from src.losses import get_companion_std

from tensorflow.keras.optimizers import Adam

cube, psf, rot_angles, table = preprocess_folder(root='./data/fake', 
												 target_folder='./data/fake/preprocessed')

dataset = create_dataset(cube, psf, rot_angles, table, batch_size=256)
model = create_embedding_model(window_size=15)

optimizer = Adam(1e-1)
model.compile(loss_fn=get_companion_std, optimizer=optimizer)
model.fit(dataset, epochs=5)
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


