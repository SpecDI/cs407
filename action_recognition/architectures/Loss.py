# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.keras import backend as K

class LossFunctions:
    def __init__(self):
        # Multiplier for positive targets for weighted_binary_cross_entropy
        self.weight = 2
    
    def weighted_binary_crossentropy(self, target, output):
        """
        Weighted binary crossentropy between an output tensor 
        and a target tensor. pos_weight is used as a multiplier 
        for the positive targets.

        Combination of the following functions:
        * keras.losses.binary_crossentropy
        * keras.backend.tensorflow_backend.binary_crossentropy
        * tf.nn.weighted_cross_entropy_with_logits
        """
        # transform back to logits
        _epsilon = K._to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))
        # compute weighted loss
        loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                        logits=output,
                                                        pos_weight=self.weight)
        return tf.reduce_mean(loss, axis=-1)

    def bce_with_label_smoothing(self, y_true, y_pred):
        """
        Binary cross entropy with label smoothing 
        """
        return tf.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.1)

    def hamming_loss(self, y_true, y_pred, tval = 0.4):
        tmp = K.abs(y_true - y_pred)
        return K.mean(K.cast(K.greater(tmp, tval), dtype = float))