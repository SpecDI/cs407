from glob import glob
import argparse
import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from colorama import Fore

import cv2
from PIL import Image
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Similarity measure")
    parser.add_argument(
        '--video_name', help = 'Name of the video to be evaluated in results dir',
        required = True
    )
    parser.add_argument(
        '--tube_name', help = 'Tube identifier for tube inspection',
        default = None
    )
    return parser.parse_args()

def compute_similarity(im1, im2, metric):
    # Compute histograms over all 3 channels (bgr)
    scores = []
    
    for i in range(3):
        hist1 = cv2.calcHist([im1],[i],None,[256],[0,256])
        hist2 = cv2.calcHist([im2],[i],None,[256],[0,256])
        
        scores.append(cv2.compareHist(hist1, hist2, metric))
    
    # Average the score over all channels
    return np.mean(scores)

def eval_tube(tube_path, test_mode = False):
    tube_frames = []
    
    # Extract and sort the frame ids
    frame_nums = sorted([int(os.path.splitext(os.path.basename(file_path))[0]) for file_path in glob(tube_path + '*.jpg')])
    for frame_num in frame_nums:
        # Compute frame path
        frame_path = tube_path + str(frame_num) + '.jpg'
        
        # Load frame
        tube_frames.append(np.asarray(Image.open(frame_path)))
    
    # List of computed pair-wise differences
    diff_list = []
    
    for i in range(len(tube_frames) - 1):
        im1 = tube_frames[i]
        im2 = tube_frames[i+1]
        
        # Compute difference
        diff_list.append(compute_similarity(im1, im2, cv2.HISTCMP_BHATTACHARYYA))
        
        # Display them only in test mode
        if test_mode:
            fig = plt.figure(figsize = (10, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(im1)
            plt.title(f"Comparing: {frame_nums[i]} and {frame_nums[i+1]}")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(im2)
            plt.title(diff_list[-1])
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.hist(im1.ravel(),256,[0,256])
            
            plt.subplot(2, 2, 4)
            plt.hist(im2.ravel(),256,[0,256])
            
            plt.show()
    
    return diff_list
        

def main(video_name, tube_name):
    video_path = f'./results/object_tracking/{video_name}/'
    
    # In test mode, evaluate only the specified tube and stop
    if tube_name is not None:
        print(eval_tube(video_path + tube_name + '/', True))
        sys.exit()
    
    # Tube-wise diff lists dictionary
    video_diff_list = {}
    
    # All computed pair-wise differences
    global_diffs = []
    
    # Order tube paths by tube id
    tube_nums = sorted([int(os.path.basename(os.path.normpath(path))) for path in glob(video_path + '**/')])
    for tube_num in tqdm(tube_nums, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        # Build tube path
        tube_path = video_path + str(tube_num) + '/'
        
        # Evaluate tube and store results
        video_diff_list[tube_num] = eval_tube(tube_path)
        global_diffs.extend(video_diff_list[tube_num])
    
    df = pd.DataFrame(columns = ['tube_path', 'tube_length', 'tube_mean', 'tube_min', 'tube_max', 'tube_std'])
    for key in video_diff_list:
        tube_diff_list = video_diff_list[key]
        
        # Frame count
        tube_length = len(tube_diff_list) + 1 
        
        # Mean
        tube_mean = np.mean(tube_diff_list)
        
        # Min/Max
        tube_min = np.min(tube_diff_list)
        tube_max = np.max(tube_diff_list)
        
        # Standard deviation
        tube_std = np.std(tube_diff_list)
        
        # Append new row
        df.loc[len(df)] = [key, tube_length, tube_mean, tube_min, tube_max, tube_std]
        
    print(df)
    print(f"\nAverage tube mean: {df['tube_mean'].mean()}")
    print(f"Average tube std: {df['tube_std'].std()}\n")
    
    print('=====GLOBAL METRICS=====')
    print(f"Global average: {np.mean(global_diffs)}")
    print(f"Global variance: {np.var(global_diffs)}")
    print(f"Global std: {np.std(global_diffs)}")

if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.video_name, args.tube_name)