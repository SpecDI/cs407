"""
Version 1.1 of the LSTM model. This is a Variational LSTM.
"""
# Keras imports 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, BatchNormalization, Bidirectional
from keras.callbacks import ModelCheckpoint

from temporal_pooling import TemporalMaxPooling2D

# Paths to be set
# TRAIN_DIR = '../../action-tubes/training/all/completed/'
# TEST_DIR = '../../action-tubes/test/'
TRAIN_DIR = '../../action-tubes/completed/'
TEST_DIR = '../../action-tubes/completed/'

# Constants to be defined
WEIGHT_FILE_NAME = '_4_1_LSTM_TempPooling'
BATCH_SIZE = 32
EPOCHS = 50

FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 8
CHANNELS = 3
CLASSES = 11

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

def cnn_lstm(input_shape, kernel_shape, pool_shape, classes):
    """
    Model definition.

    returns: Model (uncompiled)
    """
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))

    model.add(TimeDistributed(Flatten()))
    
    model.add(Dropout(0.15))
    model.add(Bidirectional(LSTM(512, recurrent_dropout=0.15, return_sequences=True)))
    model.add(TemporalMaxPooling2D())
    model.add(Dropout(0.15))

    model.add(Dense(classes, name='output', activation='sigmoid'))
    
    model.summary()
    
    return model

if __name__ == "__main__":

    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM)
    model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME)
