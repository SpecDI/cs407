# import image generator
import os
import sys

sys.path.insert(1, '../../frame_generators/')
from VideoFrameGenerator_2_1_0 import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np

datagen = ImageDataGenerator(rescale = 1.0 / 255.0)

train_dir = '../../action-tubes/training/all/completed/'
frame_num = 64
frame_length = 244
frame_width = 244
batch_size = 64

train_data = datagen.flow_from_directory(train_dir,
                                    target_size=(frame_length, frame_width),
                                    batch_size=batch_size,
                                    frames_per_step=frame_num, shuffle=True)

action_tubes, labels = train_data.next()
print(action_tubes)
print(labels)
for action_tube, label in zip(action_tubes, labels):
    
    figure = plt.figure(figsize=(10,10))
    print(train_data.class_indices)
    print(label)
    for i in range(0, 16):
        img = action_tube[i]
        print(img)
        plt.subplot(2, 8, i+1)
        plt.imshow(img)
    plt.savefig('Test_labelling.png')
    break



