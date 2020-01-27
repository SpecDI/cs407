# Imports
import os
import io
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# import image generator
import sys
sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_0_0 import ImageDataGenerator
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
train_path = '../../action-tubes/completed_amaris/'

# Constant generators
datagen = ImageDataGenerator()
train_data=datagen.flow_from_directory(train_path, target_size=(FRAME_LENGTH, FRAME_WIDTH), batch_size=BATCH_SIZE, frames_per_step=FRAME_NUM, shuffle=True)
print(train_data.next())