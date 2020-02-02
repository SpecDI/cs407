# import image generator
import sys
sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_0_0 import ImageDataGenerator

# Tensorflow imports
import tensorflow as tf
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Constant variables
FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 64
CHANNELS = 3
CLASSES = 13

"""
Python class for training and evaluating keras models.
"""
class TrainingSuite:
    def __init__(self, batch_size, epochs, train_dir, test_dir):
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_data, self.test_data = self.load_data()
    
    def load_data(self):
        datagen = ImageDataGenerator()
        train_data = datagen.flow_from_directory(self.train_dir,
                                            target_size=(FRAME_LENGTH, FRAME_WIDTH),
                                            batch_size=self.batch_size,
                                            frames_per_step=FRAME_NUM, shuffle=True)
        test_data = datagen.flow_from_directory(self.test_dir,
                                            target_size=(FRAME_LENGTH, FRAME_WIDTH),
                                            batch_size=self.batch_size,
                                            frames_per_step=FRAME_NUM, shuffle=True)  
        return train_data, test_data

    def evaluation(self, model):
        model.fit_generator(
                self.train_data,
                steps_per_epoch=11,
                epochs=self.epochs,
                validation_data=self.test_data,
                validation_steps=11
                )