import numpy as np
import os
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import sys
import math

from keras.preprocessing.image import img_to_array, ImageDataGenerator

import argparse

augmentation_prefix = 'augg'

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

def clean(path_tubes):
  print('Cleaning...')

  for im_path in list(Path(".").rglob("*.jpg")):
    im_name = str(im_path)
    if augmentation_prefix in im_name:
      os.remove(im_name)
    

def transform_image(img, tube_path, img_factor):
  datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5))

  im = img_to_array(img)
  im = im.reshape((1, ) + im.shape)

  i = 0
  for batch in datagen.flow(im, batch_size = 1, save_to_dir = tube_path, save_prefix = augmentation_prefix, save_format = 'jpg'):
    i += 1
    if i > img_factor:
      break

def augment_class(class_path, img_factor):
  print(f"Augmenting: {class_path} with factor {img_factor}")

  for tube_path in glob(class_path + '*/'):
    for im_path in glob(tube_path + '*.jpg'):
      if augmentation_prefix not in os.path.basename(os.path.normpath(im_path)):
        img = np.asarray(Image.open(im_path))
        transform_image(img, tube_path, img_factor)
      

def main(mode, path_tubes):  
  # Determine average tubes per class
  tube_counts = []
  for action_class in glob(path_tubes + '*/'):
    tube_counts.append(sum([len(files) for r, d, files in os.walk(action_class)]))

  avg_tube_size = np.mean(tube_counts).astype(int)

  i = 0
  for action_class in glob(path_tubes + '*/'):
    img_factor = math.ceil((avg_tube_size - tube_counts[i]) / tube_counts[i])

    print(f"{os.path.basename(os.path.normpath(action_class))}: {tube_counts[i]} -> {img_factor}")
    i += 1

  print(f"\nDiscovered images: {np.sum(tube_counts)}")
  print(f"Average tube size: {avg_tube_size}")

  if mode == 'clean':
    clean(path_tubes)
  elif mode == 'augmentation':
    # Augment classes with less than average
    for action_class in glob(path_tubes + '*/'):
      cnt = sum([len(files) for r, d, files in os.walk(action_class)])

      if(cnt < avg_tube_size and os.path.basename(os.path.normpath(action_class)) != 'Unknown'):
        img_factor = math.ceil((avg_tube_size - cnt) / cnt)
        augment_class(action_class, img_factor)


if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.mode, args.path)
