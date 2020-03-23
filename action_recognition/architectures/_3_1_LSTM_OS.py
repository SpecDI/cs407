"""
Version 1.1 of the LSTM model. This is a Variational LSTM.
"""
# Keras imports 
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, BatchNormalization, Input, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16


# Paths to be set
TRAIN_DIR = '../../action-tubes/training/all/completed/'
TEST_DIR = '../../action-tubes/test/'

# Constants to be defined
WEIGHT_FILE_NAME = "lstm_3_0_1"
BATCH_SIZE = 32
EPOCHS = 50

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
    input = Input(shape=input_shape)

    # CNN definition
    vgg = VGG16(input_shape=(FRAME_LENGTH, FRAME_WIDTH, CHANNELS), weights="imagenet", include_top=False)
    for layer in vgg.layers:
        if layer.name == "block5_conv3":
            break
        layer.trainable = False
    
    cnn_out = GlobalAveragePooling2D()(vgg.output)
    cnn = Model(input=vgg.input, output=cnn_out)
    
    # Combined model
    x = TimeDistributed(cnn)(input)
    x = Dropout(0.15)(x)
    x = LSTM(256, recurrent_dropout=0.15)(x)
    x = Dropout(0.15)(x)
    outputs = Dense(classes, name='output', activation='sigmoid')(x)    

    model = Model([input], outputs)
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

