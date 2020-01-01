# Author: u1611760 (Amaris Paryag)
# Imports
import os
import io
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# import image generator
import sys
sys.path.insert(1, '/dcs/16/u1611760/Year4/CS407/Frame_Generators/')
from VideoFrameGenerator_1_3_0 import ImageDataGenerator
# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPool3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint


# Constant variables
BATCH_SIZE = 32
FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 12
# Constant paths
single_label_path = '/dcs/16/u1611760/Year4/CS407/Data/single_label/'
train_path = single_label_path + 'training'
test_path = single_label_path + 'testing'

# Constant generators
datagen = ImageDataGenerator()
train_data=datagen.flow_from_directory(train_path, target_size=(FRAME_LENGTH, FRAME_WIDTH), batch_size=BATCH_SIZE, frames_per_step=FRAME_NUM, shuffle=True)
test_data=datagen.flow_from_directory(test_path, target_size=(FRAME_LENGTH, FRAME_WIDTH), batch_size=BATCH_SIZE, frames_per_step=FRAME_NUM, shuffle=True)

def cnn_3d(input_shape, kernel_shape, pool_shape, classes):
    model = Sequential()

    model.add(Conv3D(1, kernel_size=kernel_shape, input_shape=input_shape, name='conv'))
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=pool_shape, name='max_pool'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer="normal", name='fc'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, kernel_initializer="normal", name='output'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

    return model

def evaluation():
    input_shape = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)
    kernel_shape = (3, 3, 3)
    pool_shape = (2, 2, 2)
    classes = CLASSES
    epochs = 20

    model = cnn_3d(input_shape, kernel_shape, pool_shape, classes)

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_images=True,
                                                          embeddings_freq=0)
    mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit_generator(
            train_data,
            steps_per_epoch=36,
            epochs=epochs,
            validation_data=test_data,
            validation_steps=5,
            # use_multiprocessing=True,
            # max_queue_size=100,
            # workers=4,
            callbacks=[tensorboard_callback, mcp_save])

evaluation()
