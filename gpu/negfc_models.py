import tensorflow as tf

from .negfc_layers import RotateCoords, FakeCompanion, AngularDifferentialImaging
from tensorflow.keras import Model
from .losses import custom_loss


class CustomModel(tf.keras.Model):
    '''
    Custom functional model
    '''
    def compile(self, loss_fn, **kwargs):
        super(CustomModel, self).compile(**kwargs)
        self.loss_fn = loss_fn
        
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred, radius, theta = self(x, training=False)
            loss = self.loss_fn(x_pred, radius, theta)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            x_pred, radius, theta = self(x, training=False)
            loss = self.loss_fn(x_pred, radius, theta)

        return {'loss': loss}
    
    
    
def get_model(x_init, y_init, cube):
    inputs = {'cube':   tf.keras.Input(shape=cube.shape[1:]),
              'psf':    tf.keras.Input(shape=cube.shape[2:]),
              'rot_angles':    tf.keras.Input(shape=(cube.shape[1]))}

    pos_layer = RotateCoords(init_xy=(x_init, y_init))
    fake_comp_layer = FakeCompanion()
    adi_layer = AngularDifferentialImaging(cube.shape[1:], ncomp=1, reduce='median') 

    coords, radius, theta = pos_layer(inputs)
    fake = fake_comp_layer((inputs, coords))
    adi = adi_layer((fake, inputs['rot_angles']))
    out = (adi, radius, theta)
    model = CustomModel(inputs=inputs, outputs=out, name='negfc_model')
    return model