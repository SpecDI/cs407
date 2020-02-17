"""
Version 1.0 of the LSTM model.
"""
# Keras imports
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM
from keras.callbacks import ModelCheckpoint
from training import TrainingSuite

# Paths to be set
TRAIN_DIR = '../../action-tubes/completed_amaris/'
TEST_DIR = '../../action-tubes/completed_amaris/'

# Constants to be defined
WEIGHT_FILE_NAME = "lstm"
BATCH_SIZE = 32
EPOCHS = 20

FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

"""
Model definition.
"""
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

    return model

training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM)
model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)

training_suite.evaluation(model, WEIGHT_FILE_NAME)