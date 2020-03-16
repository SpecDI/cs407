"""
Version 1.1 of the LSTM model. This is a Variational LSTM.
"""
# Keras imports 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16


# Paths to be set
TRAIN_DIR = '../../action-tubes/training/all/completed/'
TEST_DIR = '../../action-tubes/test/'

# Constants to be defined
WEIGHT_FILE_NAME = "lstm_3_0_1"
BATCH_SIZE = 32
EPOCHS = 100

FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

def cnn_lstm(input_shape, kernel_shape, pool_shape, classes):
    """
    Model definition.

    returns: Model (uncompiled)
    """    
    vgg = VGG16(include_top=False)

    for layer in vgg.layers:
        if layer.name == "block5_conv3":
            break
        layer.trainable = False

    model = Sequential()
    model.add(TimeDistributed(vgg, input_shape=input_shape))
    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.15))
    model.add(LSTM(512, recurrent_dropout=0.15))
    model.add(Dropout(0.15))

    model.add(Dense(classes, name='output', activation='sigmoid'))

    model.summary()
    
    return model

if __name__ == "__main__":

    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM)
    model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME)

    # from predicting import Prediction

    # pred = Prediction(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM)
    # model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)
    # pred.probalistic_predictions(model)

