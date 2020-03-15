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
import random

from skimage.util import random_noise

from keras.preprocessing.image import img_to_array, ImageDataGenerator

import argparse

datagen = ImageDataGenerator( 
        rotation_range = 25, 
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

def gen_random_transform():
    transform_params = {
        'theta': random.uniform(0.7, 0.95),
        'tx' : random.uniform(0.7, 0.95),
        'ty' : random.uniform(0.7, 0.95),
        'shear' : random.uniform(0.7, 0.95),
        # 'zx' : random.uniform(0.9, 0.95),
        # 'zy' : random.uniform(0.9, 0.95),
        'flip_horizontal' : bool(random.getrandbits(1)),
        'flip_vertical' : bool(random.getrandbits(1)),
        'channel_shift_intencity' : 0.0,
        'brightness' : random.uniform(0.7, 0.95)
    }
    return transform_params

def augment_class(action_path, scaling_factor, prefix):
    print(f"Augmenting {os.path.basename(os.path.normpath(action_path))} with factor {scaling_factor}")

    # Iterate over each tube in given class path
    for tube in glob(action_path + '*/'):
        if not prefix in tube:
            # Generate target dir to store augmented tube
            generated_actionTubes = {}
            for i in range(scaling_factor):
                # Generate name of tube
                target_dirName = action_path + prefix + str(i) + '_' + os.path.basename(os.path.normpath(tube))
                # Save tube name with corresponding transformation
                generated_actionTubes[target_dirName] = gen_random_transform()
                os.mkdir(target_dirName)

            # Populate tubes by transforming
            image_list = glob(tube + '*.jpg')
            for im_file in sorted(image_list):
                transform_image(im_file, generated_actionTubes)
                
def transform_image(im_file, target_names):
    # Read image
    im = img_to_array(Image.open(im_file))

    # Iterate over all action generated tubes for given class
    for actionTube_key in target_names:
        # Augment image given action tube's transform
        im = datagen.apply_transform(im, target_names[actionTube_key])
        im /= 255.0
        im = random_noise(im, mode = 'gaussian', var = 0.001, clip = True)

        # Generate image name and save it
        im_name = actionTube_key + '/' + os.path.basename(os.path.normpath(im_file))
        matplotlib.image.imsave(im_name, im)

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
        
        # Print name of tube if emtpy
        if current_count == 0:
            print(f"{action_class} is EMPTY!")
            continue

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
            
        # Store factor capped at 100
        tube_map[action_key] = scale_factor if scale_factor <= 100 else 100

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