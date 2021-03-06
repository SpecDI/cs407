"""
Version 2.0 for the CNN
"""
# Keras imports 
from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, LSTM, Average, Bidirectional
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, BatchNormalization, Input
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from temporal_pooling import TemporalMaxPooling2D
from keras.regularizers import l2

# Data paths
TRAIN_DIR = '../../action-tubes/training/all/completed/'
TEST_DIR = '../../action-tubes/test/'

# Constants
WEIGHT_FILE_NAME = "_5_5_TransferLSTM_TS"
BATCH_SIZE = 8
EPOCHS = 100

FRAME_LENGTH = 80
FRAME_WIDTH = 80
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

dropout = 0.1
reg = 1e-4


def spatial_stream(input_shape):
    """
    Model definition. Has the following features:
    - Transfer Learning
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

    model = Model(inputs=[input_], outputs=x)
    model.summary()

    return model
    
def TS_CNN_LSTM(input_shape, classes):
    # Get spatial stream
    sStream = spatial_stream(input_shape)
    
    # Create LSTM temporal stream
    x = Dropout(dropout)(sStream.output)
    x = Bidirectional(LSTM(256, recurrent_dropout=dropout, return_sequences=True))(x)
    x = TemporalMaxPooling2D()(x)
    x = Dropout(dropout)(x)
    tStream = Model(inputs=sStream.input, outputs=x)
    
    # Create full 2 stream network
    spatial_average = Lambda(function=lambda x: K.mean(x, axis=1),
                   output_shape=lambda shape: (shape[0],) + shape[2:])(sStream.output)
    spatial_average = Dropout(dropout)(spatial_average)           
    spatial_out = Dense(classes, activation = 'sigmoid', kernel_regularizer=l2(reg))(spatial_average)
    
    temporal_out = Dense(classes, activation = 'sigmoid', kernel_regularizer=l2(reg))(tStream.output)
    
    averaged = Average()([spatial_out, temporal_out])
    
    model = Model(inputs=sStream.input, outputs=averaged)
    
    model.summary()
    
    return model
    

if __name__ == "__main__":
    """
    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM, preprocess_input)
    model = TS_CNN_LSTM(INPUT_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME)
    """
    from predictions_v2 import Prediction
    model = TS_CNN_LSTM(INPUT_SHAPE, CLASSES)
    preds = Prediction(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM, preprocess_input)
    preds.probablistic_predictions(model, WEIGHT_FILE_NAME)
