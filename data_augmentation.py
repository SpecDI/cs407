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

    # Iterate over each tube in given class path
    for tube in glob(action_path + '*/'):
        # If tube contains prefix delete it
        if prefix in tube:
            shutil.rmtree(tube)

def augment_class(action_path, scaling_factor, prefix):
    print(f"Augmenting {os.path.basename(os.path.normpath(action_path))} with factor {scaling_factor}")

    # Iterate over each tube in given class path
    for tube in glob(action_path + '*/'):
        if not prefix in tube:
            # Generate target dir to store augmented tube
            generated_actionTubes = []
            for i in range(scaling_factor):
                # Generate name of tube and save it
                target_dirName = action_path + prefix + str(i) + '_' + os.path.basename(os.path.normpath(tube))
                generated_actionTubes.append(target_dirName)
                os.mkdir(target_dirName)

            # Populate tubes by transforming
            image_list = glob(tube + '*.jpg')
            for im_file in sorted(image_list):
                transform_image(im_file, generated_actionTubes)
                
def transform_image(im_file, target_names):
    # Read image
    im = img_to_array(Image.open(im_file))
    
    original_shape = im.shape
    im = im.reshape((1, ) + im.shape)

    i = 0
    for batch in datagen.flow(im, batch_size = 1):
        # Stop when reached end of action tube count
        if i == len(target_names):
            break

        # Get image from data augmentator and apply some gaussian noise
        im = random_noise(batch.reshape(original_shape), mode='gaussian', clip = True)
        # Generate image name and save
        im_name = target_names[i] + '/' + os.path.basename(os.path.normpath(im_file))
        matplotlib.image.imsave(im_name, im)

        i += 1

def main(mode, path_tubes):
    # Define augmentation prefix
    prefix = 'augg_'
    
    # Dictionary that stores the number of action tubes for each class
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
    # so each class will now point some scaling factor to reach max
    for action_key in tube_map:
        scale_factor = round(maxTube_count / tube_map[action_key])
        if scale_factor == 1:
            scale_factor = 0
        tube_map[action_key] = scale_factor

    # Print out identified action classes and factors
    for action_key in tube_map:
        print(f"{os.path.basename(os.path.normpath(action_key))} -> {tube_map[action_key]}")

    # Augment/clean each action dir based on map data
    for action_key in tube_map:
        if mode == 'augmentation' and tube_map[action_key] > 0:
            augment_class(action_key, tube_map[action_key], prefix)
        elif mode == 'clean':
            clean_class(action_key, prefix)

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.mode, args.path)