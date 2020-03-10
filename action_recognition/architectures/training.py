# import image generator
import os
import sys
from datetime import datetime

# Python module import
from Metrics import MetricsAtTopK
from Loss import LossFunctions

sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_1_0 import ImageDataGenerator

# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.python.keras import backend as K

"""
Python class for training and evaluating keras models.
"""
class TrainingSuite:
    def __init__(self, batch_size, epochs, train_dir, test_dir, frame_length, frame_width, frame_num):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.frame_length = frame_length
        self.frame_width = frame_width
        self.frame_num = frame_num
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

    def evaluation(self, model, weight_file):
        metrics = MetricsAtTopK(k=2)
        losses = LossFunctions()

        model.compile(loss=losses.weighted_binary_crossentropy, optimizer='adam', metrics=['accuracy', 
                                                                                            metrics.recall_at_k, 
                                                                                            metrics.precision_at_k, 
                                                                                            metrics.f1_at_k, 
                                                                                            losses.hamming_loss])

        logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_images=True,
                                                          embeddings_freq=0)
        
        mcp_save = ModelCheckpoint('weights/' + weight_file + '.hdf5', save_best_only=True, monitor='val_f1_at_k', mode='max')

        es = EarlyStopping(monitor='val_f1_at_k', mode='max', patience=5)
        
        model.fit_generator(
                self.train_data,
                steps_per_epoch=self.train_data.samples // self.batch_size,
                epochs=self.epochs,
                validation_data=self.test_data,
                validation_steps=self.test_data.samples // self.batch_size,
                callbacks=[mcp_save, es, tensorboard_callback])
