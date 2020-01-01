# Author: u1611760 (Amaris Paryag)
# Imports
import os
import io
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# import image generator
import sys
sys.path.insert(1, '/dcs/16/u1611760/Year4/CS407/Frame_Generators/')
from TrackingVideoFrameGenerator_1_1_0 import ImageDataGenerator
# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPool3D
from keras.optimizers import SGD, RMSprop

# Constant variables
BATCH_SIZE = 32
FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 12
LABELS = ["Calling",
          "Carrying",
          "Drinking",
          "Hand Shaking",
          "Hugging",
          "Lying",
          "Pulling",
          "Reading",
          "Running",
          "Sitting",
          "Standing",
          "Walking"]
# Constant paths
# single_label_path = '/dcs/16/u1611760/Year4/CS407/Data/single_label/'
# train_path = single_label_path + 'training'
# test_path = single_label_path + 'testing'

mov_file = '1.1.1'
tracking_path = '/dcs/16/u1611760/Year4/CS407/Data/tracking_ats/deep_sort_yolov3/results/' + mov_file


# Constant generators
datagen = ImageDataGenerator()
test_data=datagen.flow_from_directory(tracking_path,
                                      target_size=(FRAME_LENGTH, FRAME_WIDTH),
                                      batch_size=BATCH_SIZE,
                                      frames_per_step=FRAME_NUM,
                                      class_mode='input',
                                      shuffle=True)

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

def predict():
    input_shape = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)
    kernel_shape = (3, 3, 3)
    pool_shape = (2, 2, 2)
    classes = CLASSES
    epochs = 20

    model = cnn_3d(input_shape, kernel_shape, pool_shape, classes)

    model.load_weights("mdl_wts.hdf5")

    for i in range(4):
        batch = test_data.next()
        ids = batch[2]
        preds = model.predict(batch[0], batch_size=32)

        # Get the max probability index of 12 classes
        max_probs = np.max(preds, axis=1)
        max_probs_index = np.argmax(preds, axis=1)
        # map the index to the corresponding class
        actions = [LABELS[j] for j in max_probs_index]
        # stitch the action class label to the ids
        predictions = pd.DataFrame({
            'ID': ids,
            'Action': actions,
            'Probability': max_probs
        })
        # save this in a file.
        predictions.to_csv('predictions.txt',
                            header=None,
                            index=None,
                            sep=',',
                            mode='a')
predict()
