import tensorflow as tf

from .negfc_layers import RotateCoords, \
                          FakeCompanion, \
                          AngularDifferentialImaging, \
                          MoveScalePSF, \
                          FakeInjection
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
            x_pred, radius, theta = self(x, training=True)
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

class CustomModelNew(tf.keras.Model):
    '''
    Custom functional model
    '''
    def compile(self, loss_fn, **kwargs):
        super(CustomModelNew, self).compile(**kwargs)
        self.loss_fn = loss_fn
        
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            fake_comp = self(x, training=True)
            loss = self.loss_fn(y, fake_comp)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            fake_comp, xypos = self(x, training=True)
            loss = self.loss_fn(y, fake_comp)
        return {'loss': loss}
    
def get_model_new(init_xy, init_flux, cube_shp):
    inputs = {'psf': tf.keras.Input(shape=cube_shp[1:], batch_size=1),
              'rot_angles':    tf.keras.Input(shape=(cube_shp[0]), batch_size=1)}
    
    move_scale_layer = MoveScalePSF(init_xy, init_flux, name='MoveScale')
    
    # === pipeline === 
    output = move_scale_layer(inputs)
    
    # === MODEL ====
    model = CustomModelNew(inputs=inputs, outputs=output, name='negfc_model')
    return model

class CustomModelAngle(tf.keras.Model):
    '''
    Custom functional model
    '''
    def compile(self, loss_fn, **kwargs):
        super(CustomModelAngle, self).compile(**kwargs)
        self.loss_fn = loss_fn
        
    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            fake_comp = self(x, training=True)
            loss = self.loss_fn(y, fake_comp)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}

    @tf.function
    def test_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            fake_comp, xypos = self(x, training=True)
            loss = self.loss_fn(y, fake_comp)
        return {'loss': loss}

def get_model_angle(init_radius, init_theta, init_flux, 
                    cube_shp):
    inputs = {'psf': tf.keras.Input(shape=cube_shp[1:], batch_size=1),
              'rot_angles':    tf.keras.Input(shape=(cube_shp[0]), batch_size=1)}
    
    move_scale_layer = FakeInjection(init_radius,
                                     init_theta, 
                                     init_flux, 
                                     name='AngleBasedModel')
    
    # === pipeline === 
    output = move_scale_layer(inputs)
    
    # === MODEL ====
    model = CustomModelAngle(inputs=inputs, outputs=output, name='negfc_model')
    return model