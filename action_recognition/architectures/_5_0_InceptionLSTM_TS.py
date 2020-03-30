"""
Version 2.0 for the CNN
"""
# Keras imports 
from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, LSTM, Average, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, BatchNormalization, Input
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from temporal_pooling import TemporalMaxPooling2D

# Data paths
TRAIN_DIR = '../../action-tubes/training/all/completed/'
TEST_DIR = '../../action-tubes/test/'

# Constants
WEIGHT_FILE_NAME = "_5_0_CNN_LSTM"
BATCH_SIZE = 8
EPOCHS = 50

FRAME_LENGTH = 244
FRAME_WIDTH = 244
FRAME_NUM = 32
CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

def spatial_stream(input_shape):
    """
    Model definition. Has the following features:
    - Transfer Learning
    """
    inception = InceptionV3(input_shape=(FRAME_LENGTH, FRAME_WIDTH, CHANNELS),
                        weights="imagenet",
                        include_top=False,
                        pooling='avg')

    for layer in inception.layers:
        layer.trainable = False 

    cnn = Model(inputs=inception.input, outputs=inception.output)
    #cnn.summary()

    input_ = Input(shape=input_shape)
    x = TimeDistributed(cnn)(input_)

    model = Model(inputs=[input_], outputs=x)
    #model.summary()

    return model
    
def TS_CNN_LSTM(input_shape, classes):
    # Get inception spatial stream
    sStream = spatial_stream(input_shape)
    
    # Create LSTM temporal stream
    x = Dropout(0.15)(sStream.output)
    x = Bidirectional(LSTM(128, recurrent_dropout=0.15, return_sequences=False))(x)
    x = Bidirectional(LSTM(128, recurrent_dropout=0.15, return_sequences=False))(x)
    x = Bidirectional(LSTM(128, recurrent_dropout=0.15, return_sequences=True))(x)
    x = TemporalMaxPooling2D()(x)
    x = Dropout(0.15)(x)
    tStream = Model(inputs=sStream.input, outputs=x)
    
    # Create full 2 stream network
    spatial_average = Lambda(function=lambda x: K.mean(x, axis=1),
                   output_shape=lambda shape: (shape[0],) + shape[2:])(sStream.output)
    spatial_out = Dense(classes, activation = 'sigmoid')(spatial_average)
    
    temporal_out = Dense(classes, activation = 'sigmoid')(tStream.output)
    
    averaged = Average()([spatial_out, temporal_out])
    
    model = Model(inputs=sStream.input, outputs=averaged)
    
    model.summary()
    
    return model
    

if __name__ == "__main__":

    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM, preprocess_input)
    model = TS_CNN_LSTM(INPUT_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME, gen_logs = False)
