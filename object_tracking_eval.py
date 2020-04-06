from glob import glob
import argparse
from PIL import Image
import os
import numpy as np

from scipy.stats import ks_2samp
from scipy.stats import norm
from scipy.stats import wasserstein_distance as wd

import cv2

from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Similarity measure")
    parser.add_argument(
        '--video_name', help = 'Name of the video to be evaluated in results dir',
        required = True
    )
    return parser.parse_args()

def compute_similarity(im1, im2, metric):
    # Compute the image histograms
    # im1_hist, im1_edges = np.histogram(im1.ravel(), 64)
    # im2_hist, im2_edges = np.histogram(im2.ravel(), 64)
    
    hist1 = cv2.calcHist([im1],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([im2],[0],None,[256],[0,256])

    diff = cv2.compareHist(hist1, hist2, metric)
    
    return diff

def eval_tube(tube_path):
    print(f'Processing {tube_path}')
    tube_frames = []
    
    frame_nums = sorted([int(os.path.splitext(os.path.basename(file_path))[0]) for file_path in glob(tube_path + '*.jpg')])
    for frame_num in frame_nums:
        # Compute frame path
        frame_path = tube_path + str(frame_num) + '.jpg'
        
        # Load frame
        tube_frames.append(np.asarray(Image.open(frame_path)))
    
    
    # Iterate over all frames -1
    # Compute similarity
    running_sum_chi = 0
    running_sum_bhat = 0
    running_sum_inter = 0
    
    for i in range(len(tube_frames) - 1):
        im1 = tube_frames[i]
        im2 = tube_frames[i+1]
        
        # Compute difference
        chi_score = compute_similarity(im1, im2, cv2.HISTCMP_CHISQR)
        bhat_score = compute_similarity(im1, im2, cv2.HISTCMP_BHATTACHARYYA)
        inter_score = compute_similarity(im1, im2, cv2.HISTCMP_INTERSECT)
        
        chi_factor = chi_score / (running_sum_chi / i) if i != 0 else 1
        bhat_factor = bhat_score / (running_sum_bhat / i) if i != 0 else 1
        inter_factor = inter_score / (running_sum_inter / i) if i != 0 else 1
        
        running_sum_chi += chi_score
        running_sum_bhat += bhat_score
        running_sum_inter += inter_score
        
        print('CHI: ', round(chi_score, 2), '|', round(chi_factor, 2))
        print('BHAT: ', round(bhat_score, 2), '|', round(bhat_factor, 2))
        print('INTER: ', round(inter_score, 2), '|', round(inter_factor, 2))
        print('')
        
        # Display them
        fig = plt.figure(figsize = (10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(im1)
        plt.title(frame_nums[i])
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(im2)
        plt.title(frame_nums[i+1])
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.hist(im1.ravel(),256,[0,256])
        
        plt.subplot(2, 2, 4)
        plt.hist(im2.ravel(),256,[0,256])
        
        plt.show()
        

def main(video_name):
    video_path = f'./results/object_tracking/{video_name}/**/'
    
    eval_tube('./results/object_tracking/webServer_inputVideo/663/')
    # for tube_path in sorted(glob(video_path)):
    #     eval_tube(tube_path)


if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.video_name)