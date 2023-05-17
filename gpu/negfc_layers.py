import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf 
import multiprocessing as mp
import numpy as np

from tensorflow.keras.layers import Layer
from .fake_comp import create_patch, pca_tf, rotate_cube    
        
        
class RotateCoords(Layer):
    def __init__(self, init_xy):
        super(RotateCoords, self).__init__()
        self.init_x  = tf.cast(init_xy[0], tf.float32)
        self.init_y  = tf.cast(init_xy[1], tf.float32)
        
    def build(self, input_shape):  # Create the state of the layer (weights)
        init_x = tf.constant(self.init_x, shape=(1), dtype=tf.float32)
        self.x = tf.Variable(shape=(1),
                             initial_value=init_x,
                             trainable=True, 
                             name='xcord')
        
        init_y = tf.constant(self.init_y, shape=(1), dtype=tf.float32)
        self.y = tf.Variable(shape=(1),
                             initial_value=init_y,
                             trainable=True, 
                             name='ycord')
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        '''
        inputs is a dictonary with key 'cube' and value a tensor of dimension (n_frames, width, height)
        '''
        n_frames = tf.shape(inputs['cube'])[-3]
        width    = tf.cast(tf.shape(inputs['cube'])[-2], dtype=tf.float32)
        height   = tf.cast(tf.shape(inputs['cube'])[-1], dtype=tf.float32)

        width_s  = width/tf.constant(2., dtype=tf.float32)
        height_s = height/tf.constant(2., dtype=tf.float32)
        
        x = self.x - width_s   # origin in the middle of the frame
        y = self.y - height_s # origin in the middle of the frame

        radius = tf.sqrt(tf.pow(x, 2)+tf.pow(y, 2))
        
        angle = tf.atan2(y, x) # radians
        angle = angle/tf.constant(np.pi)*180.  # degrees
        theta = tf.math.mod(angle, 360.) # bound up to 1 circumference      
        
        rot_theta = theta - inputs['rot_angles'] # rotate angles 
        rot_theta = tf.experimental.numpy.deg2rad(rot_theta)
        
        x_s = tf.multiply(radius, tf.cos(rot_theta)) #+ width_s
        y_s = tf.multiply(radius, tf.sin(rot_theta)) #+ height_s

        shift_indices = tf.stack([x_s, y_s], axis=2)
        
        return shift_indices, radius, theta
    
    
class FakeCompanion(Layer):
    '''
    Inject a fake companion
    '''
    def __init__(self, init_flux=4):
        super(FakeCompanion, self).__init__()
        self.init_flux = init_flux
        
    def build(self, input_shape):  # Create the state of the layer (weights)
        init_flux = tf.constant(self.init_flux, shape=(1), dtype=tf.float32)
        self.flux = tf.Variable(shape=(1),
                                initial_value=init_flux,
                                trainable=True, 
                                name='flux')
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        '''
        inputs is a dictonary with key 'psf' and value a tensor of dimension (width, height)
        '''
        coords = inputs[1]
        
        n_frames = tf.shape(inputs[0]['cube'])[1]
        
        cube_patch = tf.expand_dims(inputs[0]['psf'], 1)
        cube_patch = tf.tile(cube_patch, [1, n_frames, 1, 1])
        scaled_patch = tf.multiply(cube_patch, -self.flux)
        scaled_patch = tf.expand_dims(scaled_patch, -1)
        
        def fn(x1, x2):
            return tfa.image.translate(x1, x2)
        
        fake_comp =tf.map_fn(lambda x: fn(x[0], x[1]), 
                             (scaled_patch, coords),
                             dtype=(tf.float32)
                            )
        fake_comp = tf.squeeze(fake_comp)
        output = fake_comp + inputs[0]['cube']
        return tf.expand_dims(output, -1) 
    
class AngularDifferentialImaging(Layer):
    '''
    AngularDifferentialImaging
    '''
    def __init__(self, out_size, ncomp=1, reduce='median', on='fake'):
        super(AngularDifferentialImaging, self).__init__()
        self.ncomp = ncomp
        self.reduce = reduce
        self.on = on
        self.out_size =out_size
            
    def call(self, inputs):  # Defines the computation from inputs to outputs
        '''
        inputs is a cube (n_frames, width, height)
        '''
        cube = inputs[0]
        rot_angles = inputs[1]
        
        n_frames = tf.shape(cube)[1]
        width    = tf.cast(tf.shape(cube)[-3], dtype=tf.int32)
        height   = tf.cast(tf.shape(cube)[-2], dtype=tf.int32)
          
        def perform_pca(batch_cube, batch_angles):        
            batch_cube = tf.reshape(batch_cube, [n_frames, width, height])

            res, cube_rec = pca_tf(batch_cube, self.out_size, ncomp=self.ncomp)
                        
            res_derot = tf.map_fn(lambda x: rotate_cube(x[0], x[1], derotate='tf'), 
                                 (res, batch_angles), dtype=(tf.float32))
            res_derot = tf.reshape(res_derot, [n_frames, width, height])
            collapsed = tfp.stats.percentile(res_derot, 50.0, 
                                             interpolation='midpoint', axis=0)
            collapsed = tf.reshape(collapsed, [width, height])
        
            return collapsed
        
        collapsed = tf.map_fn(lambda x: perform_pca(x[0], x[1]), 
                              (cube, rot_angles), dtype=(tf.float32))

        return collapsed
    
    
class MoveScalePSF(Layer):
    def __init__(self, init_xy, init_f, **kwargs):
        super(MoveScalePSF, self).__init__(**kwargs)
        self.init_x  = tf.cast(init_xy[:, 0], tf.float32)
        self.init_y  = tf.cast(init_xy[:, 1], tf.float32)
        self.init_f  = tf.cast(init_f, tf.float32)
        
        self.n_candidates = tf.shape(self.init_x)[0]
        
    def build(self, input_shape):  # Create the state of the layer (weights)
        init_x = tf.constant(self.init_x, 
                             shape=(self.n_candidates, 1), 
                             dtype=tf.float32)
        self.x = tf.Variable(initial_value=init_x,
                             trainable=True, 
                             name='xcoord')

        init_y = tf.constant(self.init_y, 
                             shape=(self.n_candidates, 1), 
                             dtype=tf.float32)
        self.y = tf.Variable(initial_value=init_y,
                             trainable=True, 
                             name='ycoord')

        init_f = tf.constant(self.init_f, 
                             shape=(self.n_candidates, 1), 
                             dtype=tf.float32)
        self.flux = tf.Variable(initial_value=init_f,
                                trainable=True, 
                                name='flux')
    @tf.function
    def call(self, inputs):
        n_frames = tf.shape(inputs['rot_angles'])[1]
        width    = tf.cast(tf.shape(inputs['psf'])[-2], dtype=tf.float32)
        height   = tf.cast(tf.shape(inputs['psf'])[-1], dtype=tf.float32)
        
        width_s  = width/tf.constant(2., dtype=tf.float32)
        height_s = height/tf.constant(2., dtype=tf.float32)
        
        x_c = self.x - width_s # origin in the middle of the frame 
        y_c = self.y - height_s # origin in the middle of the frame
        
        radius = tf.sqrt(tf.pow(x_c, 2)+tf.pow(y_c, 2))
        
        angle = tf.atan2(y_c, x_c) # radians
        angle = angle/tf.constant(np.pi)*180.  # degrees
        theta = tf.math.mod(angle, 360.) # bound up to 1 circumference  
            
        rot_theta = theta - inputs['rot_angles'] # rotate angles 
        rot_theta = tf.experimental.numpy.deg2rad(rot_theta)
        
        x_s = tf.multiply(radius, tf.cos(rot_theta)) #+ width_s
        y_s = tf.multiply(radius, tf.sin(rot_theta)) #+ height_s

        shift_indices = tf.stack([x_s, y_s], axis=2)
        
        cube_patch = tf.expand_dims(inputs['psf'], 1)
        cube_patch = tf.tile(cube_patch, [self.n_candidates, n_frames, 1, 1])
        
        def fn(cube, xy_translations):
            return tfa.image.translate(tf.expand_dims(cube, -1), xy_translations)
            
        
        patch = tf.map_fn(lambda x: fn(x[0], x[1]), 
                             (cube_patch, shift_indices),
                              fn_output_signature=(tf.float32),
                              parallel_iterations=mp.cpu_count()//2,
                         name='translate')      
        
        fluxes = tf.tile(self.flux, [1, n_frames])
        fluxes = tf.reshape(fluxes, [self.n_candidates, n_frames, 1, 1, 1])
        
        partial_cords = tf.stack([self.x, self.y], 1)
        partial_cords = tf.reshape(partial_cords, [self.n_candidates, 2])
        
        return tf.multiply(patch, fluxes), partial_cords