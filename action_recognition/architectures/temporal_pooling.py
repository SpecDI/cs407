import tensorflow as tf
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np

class TemporalMaxPooling2D(Layer):
    
    """
    Acknowledgements
    
    The implemented temporal pooling algorithm is based
    on the community supplied implementation available at
    
    https://gist.github.com/nigeljyng/881ae30e7c35ca2b77f6975e50736493
    
    This was also created as a response to the work of Tan et al.,
    on LSTM-based deep learning models for nonfactoid answer selection
    https://arxiv.org/pdf/1511.04108.pdf
    """
    
    def __init__(self, **kwargs):
        super(TemporalMaxPooling2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim = 3)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def call(self, x, mask = None):
        if mask is None:
            mask = K.sum(K.ones_like(x), axis = -1)
            
        mask = K.expand_dims(mask, axis = -1)
        mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
        masked_data = tf.where(K.equal(mask, K.zeros_like(mask)), K.ones_like(x) * -np.inf, x)
        return K.max(masked_data, axis = 1)
    
    def compute_mask(self, input, mask):
        return None