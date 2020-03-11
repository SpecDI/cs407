import numpy as np
import os
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from pathlib import Path
import sys
import math
import shutil

from skimage.util import random_noise

from keras.preprocessing.image import img_to_array, ImageDataGenerator

import argparse

datagen = ImageDataGenerator( 
        rotation_range = 15, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        brightness_range = (0.5, 1.5),
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        rescale = 1.0 / 255.0)

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument(
        "--mode", help="Mode = clean will result in deleting all augmented tubes. Mode = augmentation will result in augmentation. No argument will print stats"
    )
    parser.add_argument(
        "--path", help="Path to action-tubes folder to be augmented",
        required = True
    )
    return parser.parse_args()

def clean_class(action_path, prefix):
    print('Cleaning: ', os.path.basename(os.path.normpath(action_path)))
    for tube in glob(action_path + '*/'):
        if prefix in tube:
            shutil.rmtree(tube)

def augment_class(action_path, scaling_factor, prefix):
    print(f"Augmenting {os.path.basename(os.path.normpath(action_path))} with factor {scaling_factor}")

    for tube in glob(action_path + '*/'):
        if not prefix in tube:
            # Generate target augmentated tube
            generated_actionTubes = []
            for i in range(scaling_factor):
                target_dirName = action_path + prefix + str(i) + '_' + os.path.basename(os.path.normpath(tube))
                generated_actionTubes.append(target_dirName)
                os.mkdir(target_dirName)

            # Populate tubes by transforming
            image_list = glob(tube + '*.jpg')
            for im_file in sorted(image_list):
                transform_image(im_file, generated_actionTubes)
                
def transform_image(im_file, target_names):
    im = img_to_array(Image.open(im_file))
    
    original_shape = im.shape
    im = im.reshape((1, ) + im.shape)

    i = 0
    for batch in datagen.flow(im, batch_size = 1):
        im = random_noise(batch.reshape(original_shape), mode='gaussian', clip = True)

        im_name = target_names[i] + '/' + os.path.basename(os.path.normpath(im_file))
        matplotlib.image.imsave(im_name, im)

        i += 1
        if i == len(target_names):
            break

def main(mode, path_tubes):
    # Define augmentation prefix
    prefix = 'augg_'

    tube_map = {}
    maxTube_count = 0
    # Iterate over all action classes
    for action_class in glob(path_tubes + '*/'):
        # Get the number of tubes in a given class
        current_count = len([x[0] for x in os.walk(action_class)]) - 1 # don't count . dir
        
        # Insert tube count in map
        tube_map[action_class] = current_count
        
        # Update max count
        if current_count > maxTube_count:
            maxTube_count = current_count
    
    # Update count to scaling factor
    for action_key in tube_map:
        tube_map[action_key] = math.ceil(maxTube_count / tube_map[action_key])

    # Print out identified action classes and factors
    for action_key in tube_map:
        print(f"{os.path.basename(os.path.normpath(action_key))} -> {tube_map[action_key]}")

    # Augment each action dir based on map data
    for action_key in tube_map:
        if mode == 'augmentation':
            augment_class(action_key, tube_map[action_key], prefix)
        elif mode == 'clean':
            clean_class(action_key, prefix)
    
    # print('Max action tube count: ', maxTube_count)
    # print(tube_map)

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.mode, args.path)