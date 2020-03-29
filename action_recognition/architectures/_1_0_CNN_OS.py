"""
Version 2.0 for the CNN
"""
# Keras imports 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, BatchNormalization, Input
from tensorflow.python.keras import backend as K
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

# Data paths
# TRAIN_DIR = '../../action-tubes/training/all/completed/'
# TEST_DIR = '../../action-tubes/test/'
TRAIN_DIR = '../../action-tubes/completed/'
TEST_DIR = '../../action-tubes/completed/'

# Constants
WEIGHT_FILE_NAME = "_1_0_cnn_OS"
BATCH_SIZE = 8
EPOCHS = 100

FRAME_LENGTH = 244
FRAME_WIDTH = 244
FRAME_NUM = 16
CHANNELS = 3
CLASSES = 11

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

def cnn(input_shape, kernel_shape, pool_shape, classes):
    """
    Model definition. Has the following features:
    - Transfer Learning
    """
    embedded_dims = 512
    resnet = ResNet50(input_shape=(FRAME_LENGTH, FRAME_WIDTH, CHANNELS),
                        weights="imagenet",
                        include_top=False,
                        pooling='avg')

    for layer in resnet.layers:
        layer.trainable = False 

    cnn = Model(inputs=resnet.input, outputs=resnet.output)
    cnn.summary()

    input_ = Input(shape=input_shape)
    x = TimeDistributed(cnn)(input_)
    x = Lambda(function=lambda x: K.mean(x, axis=1),
                   output_shape=lambda shape: (shape[0],) + shape[2:])(x)
    outputs = Dense(classes, name='output', activation='sigmoid')(x)

    model = Model(inputs=[input_], outputs=outputs)
    model.summary()

    return model

if __name__ == "__main__":

    from training import TrainingSuite

    training_suite = TrainingSuite(BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR, FRAME_LENGTH, FRAME_WIDTH, FRAME_NUM, preprocess_input)
    model = cnn(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)

    training_suite.evaluation(model, WEIGHT_FILE_NAME)
