"""
Version 1.0 for the CNN-LSTM. 
"""
# Keras imports 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, BatchNormalization
from tensorflow.python.keras import backend as K

# Data paths
TRAIN_DIR = '../../action-tubes/training/all/completed/'
TEST_DIR = '../../action-tubes/test/'

# Constants
WEIGHT_FILE_NAME = "_1_2_lstm_OS"
BATCH_SIZE = 32
EPOCHS = 100

FRAME_LENGTH = 200
FRAME_WIDTH = 200
FRAME_NUM = 16
CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

def cnn_lstm(input_shape, kernel_shape, pool_shape, classes):
    """
    Model definition. Has the following features:
    - Temporal Averaging
    returns: Model (uncompiled)
    """

    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape), input_shape=input_shape))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape)))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape)))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, return_sequences=True))
    # Temporal Mean Pooling
    model.add(Lambda(function=lambda x: K.mean(x, axis=1),
                   output_shape=lambda shape: (shape[0],) + shape[2:]))
    model.add(Dense(classes, name='output', activation='sigmoid'))
    model.summary() 
    return model

if __name__ == "__main__":

    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM)
    model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME)
