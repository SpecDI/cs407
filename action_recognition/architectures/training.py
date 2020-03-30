# import image generator
import os
import sys
from datetime import datetime

# Python module import
from Metrics import MetricsAtTopK, RankMetrics
from Loss import LossFunctions
#from Optimisers import AdaBound
sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_1_0 import ImageDataGenerator

# Tensorflow imports
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adagrad

"""
Python class for training and evaluating keras models.
"""
class TrainingSuite:
    def __init__(self, batch_size, epochs, train_dir, test_dir, frame_length, frame_width, frame_num, preprocessing_function=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.frame_length = frame_length
        self.frame_width = frame_width
        self.frame_num = frame_num
        self.preprocessing_function = preprocessing_function
        self.train_data, self.test_data = self._load_data()
    
    def _load_data(self):
        datagen = ImageDataGenerator(preprocessing_function=self.preprocessing_function)
        train_data = datagen.flow_from_directory(self.train_dir,
                                            target_size=(self.frame_length, self.frame_width),
                                            batch_size=self.batch_size,
                                            frames_per_step=self.frame_num, shuffle=True)
        test_data = datagen.flow_from_directory(self.test_dir,
                                            target_size=(self.frame_length, self.frame_width),
                                            batch_size=self.batch_size,
                                            frames_per_step=self.frame_num, shuffle=True)  
        return train_data, test_data

    def evaluation(self, model, weight_file, gen_logs = True):
        metrics = MetricsAtTopK(k=3)
        rank_metrics = RankMetrics()
        losses = LossFunctions()
        optimiser = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss="binary_crossentropy", optimizer=optimiser, metrics=[losses.hamming_loss,
                                                                                rank_metrics.one_error,
                                                                                metrics.recall_at_k, 
                                                                                metrics.precision_at_k, 
                                                                                metrics.f1_at_k])
        
        mcp_save = ModelCheckpoint('weights/' + weight_file + '.hdf5', save_best_only=True, monitor='val_f1_at_k', mode='max')
        es = EarlyStopping(monitor='val_f1_at_k', mode='max', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='val_f1_at_k', mode='max', factor=0.2, patience=5, verbose=1)
        
        callbacks = [mcp_save, es, reduce_lr]

        if gen_logs:
            logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,
                                                            histogram_freq=1,
                                                            write_graph=True,
                                                            write_images=True,
                                                            embeddings_freq=0)
            callbacks.append(tensorboard_callback)
        
        
        model.fit_generator(
                self.train_data,
                steps_per_epoch=self.train_data.samples // self.batch_size,
                epochs=self.epochs,
                validation_data=self.test_data,
                validation_steps=self.test_data.samples // self.batch_size,
                validation_freq=5,
                callbacks=callbacks)
