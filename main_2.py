import tensorflow as tf
import wandb

from src.preprocess import preprocess_folder
from src.format_data import create_dataset
from src.model import create_embedding_model, create_flux_model, create_convnet
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from src.losses import wrapper, keep_back

from tensorflow.keras.optimizers import Adam
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


WEIGHTS_FOLDER = './logs/hp_results'
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)



# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
    'name': 'ASTROMER_I',
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'epoch/loss'},
    'early_terminate':{
      'type': 'hyperband',
      'min_iter': 10},
    'parameters': {
        'learning_rate': {'max': 1e-1, 'min': 1e-4},
        'back_q': {'values': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
        'window_size': {'values': [10, 15, 20, 25, 30, 50, 100]},
        'decay_factor': {'values': [1, 2, 3, 5, 8, 13]}
    }
}


def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        cube, psf, rot_angles, table = preprocess_folder(root='./data/fake', 
                                                         target_folder='./data/fake/preprocessed')

        window_size = 30
        table = table[table['snr'] > 1]
        dataset = create_dataset(cube, 
                                 psf, 
                                 rot_angles, 
                                 table, 
                                 window_size=config.window_size, 
                                 batch_size=1000, 
                                 repeat=1,
                                 back_q=config.back_q)

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

# =====================================================================================
# ===== WandB =========================================================================
# =====================================================================================
sweep_id = wandb.sweep(sweep_conf, project="hp-negfc")
wandb.agent(sweep_id, function=sweep_train, count=100)