"""
TODO: 
- Change the metric for evaluating multi-action label (maybe use the metric from the paper)
- Get tensorboard working.

Image generator
- Change how we pad frames
- Change how we deal with data augmentation
- Change how we deal with frames of different sizes

Model
- Play around with architecture
"""

# Imports
import os
import io
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# import image generator
import sys
sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_1_0 import ImageDataGenerator
# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Keras imports
from tensorflow.python.keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import hamming_loss

# Constant variables
BATCH_SIZE = 32
FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 13
# Constant paths
train_path = '../../action-tubes/training/all/completed'
test_path = '../../action-tubes/test'

# Constant generators
datagen = ImageDataGenerator()
train_data=datagen.flow_from_directory(train_path, target_size=(FRAME_LENGTH, FRAME_WIDTH), batch_size=BATCH_SIZE, frames_per_step=FRAME_NUM, shuffle=True)
test_data=datagen.flow_from_directory(test_path, target_size=(FRAME_LENGTH, FRAME_WIDTH), batch_size=BATCH_SIZE, frames_per_step=FRAME_NUM, shuffle=True)

def cnn_lstm(input_shape, kernel_shape, pool_shape, classes):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape, activation='relu')))

    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(100))
    model.add(Dropout(0.5))

    model.add(Dense(classes, kernel_initializer="normal", name='output'))
    model.add(Activation('softmax'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=[hamming_loss, 'accuracy'])

    return model

def evaluation():
    # Change FRAME_NUM to None if you want varying action tube length across batches.
    input_shape = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)
    kernel_shape = (3, 3)
    pool_shape = (2, 2)
    classes = CLASSES
    epochs = 1
    train_steps = train_data.samples // BATCH_SIZE
    test_steps = test_data.samples // BATCH_SIZE

    model = cnn_lstm(input_shape, kernel_shape, pool_shape, classes)

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_images=True,
                                                          embeddings_freq=0)
    mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        
    model.fit_generator(
            train_data,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=test_data,
            validation_steps=test_steps,
            # use_multiprocessing=True,
            # max_queue_size=100,
            # workers=4,
            callbacks = [tensorboard_callback, mcp_save]
            )

    metrics = model.evaluate_generator(test_data, steps = test_steps)
    for i in range(len(metrics)):
        print(f"{model.metrics_names[i]}: {metrics[i]}")
    

def hamming_loss(y_true, y_pred, tval = 0.4):
    tmp = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.greater(tmp, tval), dtype = float))

    

evaluation()