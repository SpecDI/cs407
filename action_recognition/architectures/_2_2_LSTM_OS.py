"""
Version 2.0 for the CNN-LSTM. 
"""
# Keras imports 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM, BatchNormalization, Input
from tensorflow.python.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.regularizers import l2

# Data paths
TRAIN_DIR = '../data/action_tubes/Training/'
TEST_DIR = '../data/action_tubes/Test/'

# Constants
WEIGHT_FILE_NAME = "_2_2_LSTM_OS"
BATCH_SIZE = 8
EPOCHS = 100

FRAME_LENGTH = 80
FRAME_WIDTH = 80
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 13

INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

dropout = 0.1
reg = 1e-4

def cnn_lstm(input_shape, classes):
    """
    Model definition. Has the following features:
    - Transfer Learning
    - Temporal Averaging
    returns: Model (uncompiled)
    """
    base = VGG16(input_shape=(FRAME_LENGTH, FRAME_WIDTH, CHANNELS),
                        weights="imagenet",
                        include_top=False)

    for layer in base.layers:
        layer.trainable = False 
    
    x = Flatten()(base.output)
    x = Dropout(dropout)(x)
    x = Dense(512, activation = 'relu', kernel_regularizer=l2(reg))(x)
    
    cnn = Model(inputs=base.input, outputs=x)
    cnn.summary()

    input_ = Input(shape=input_shape)
    x = TimeDistributed(cnn)(input_)
    
    x = Dropout(dropout)(x)
    x = LSTM(256, recurrent_dropout=dropout, return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    outputs = Dense(classes, name='output', activation='sigmoid', kernel_regularizer=l2(reg))(x)    

    model = Model(inputs=[input_], outputs=outputs)
    model.summary()
    return model

if __name__ == "__main__":

    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM, preprocess_input)
    model = cnn_lstm(INPUT_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME)
