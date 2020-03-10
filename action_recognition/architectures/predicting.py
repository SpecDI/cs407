import os
import sys
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Python module import
from Metrics import MetricsAtTopK
from Loss import LossFunctions

# import image generator
sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_1_0 import ImageDataGenerator

# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.python.keras import backend as K

"""
Python class for getting predictions of keras models.
"""
class Prediction:
    def __init__(self, batch_size, epochs, train_dir, test_dir, frame_length, frame_width, frame_num):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.frame_length = frame_length
        self.frame_width = frame_width
        self.frame_num = frame_num
        self.actions = sorted(['Unknown', 'Sitting', 'Lying', 'Drinking', 'Calling', 'Reading', 'Handshaking', 'Running', 'Pushing_Pulling', 'Walking', 'Hugging', 'Carrying', 'Standing'])
        self.train_data, self.test_data = self._load_data()
    
    def _load_data(self):
        datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
        train_data = datagen.flow_from_directory(self.train_dir,
                                            target_size=(self.frame_length, self.frame_width),
                                            batch_size=self.batch_size,
                                            frames_per_step=self.frame_num, shuffle=True)
        test_data = datagen.flow_from_directory(self.test_dir,
                                            target_size=(self.frame_length, self.frame_width),
                                            batch_size=self.batch_size,
                                            frames_per_step=self.frame_num, shuffle=True)  
        return train_data, test_data

    def predict_with_uncertainty(self, model, x, n_iter=10):
        """
        Makes predictions using the MC dropout method. 

        param model: Model with dropout layers.
        param x: Input for prediction.

        returns: predictions, uncertainty of prediction
        """

        f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
        
        result = np.zeros((n_iter, ) + (1, 32, 64, 13))

        for iter in range(n_iter):
            result[iter] = f([x,1])

        prediction = result.mean(axis=0)
        uncertainty = result.var(axis=0)
    
        return prediction, uncertainty

        # return result
    
    def probalistic_predictions(self, model): 
        """
        Probablistic prediction evaluation
        """   
        metrics = MetricsAtTopK(k=2)
        losses = LossFunctions()

        model.load_weights('weights/lstm_1_4_1.hdf5')

        next_ = self.train_data.next()
        test_item = next_[0]
        test_y = next_[1]

        print(test_y.shape)

        # MC dropout predictions
        preds, uncertainty = self.predict_with_uncertainty(model, test_item)

        # first frame of first action tube
        for j in range(6):
            f_test_item = test_item[0][j*10]
            f_test_y = test_y[0]

            f_preds = preds[0][0][j*10]
            f_uncert = uncertainty[0][0][j*10]
            
            plt.subplot(2, 1, 1)
            plt.title(str(self.actions) + "\n" + str(f_test_y))
            plt.imshow(f_test_item.astype(np.uint8))
            plt.subplot(2, 1, 2)

            print("Frame: " + str(j*10))
            for i in range(len(self.actions)):
                mu = f_preds[i]
                sigma = math.sqrt(f_uncert[i])

                print("\t " + str(self.actions[i]) + ": Mean: " + str(mu) + "Std: " + str(sigma))

                x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                y = stats.norm.pdf(x, mu, sigma)
                plt.plot(x, y, label=self.actions[i])
                plt.fill_between(x, 0, y, alpha=0.5)

            plt.legend()
            plt.show()