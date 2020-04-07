from operator import truediv, add

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


# Implemente KS with channels
# Try a local descriptor e.g., SIFT and SURF


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
    
    scores = []
    
    for i in range(3):
        hist1 = cv2.normalize(cv2.calcHist([im1],[i],None,[256],[0,256]), None, 0, 1, cv2.NORM_MINMAX)
        hist2 = cv2.normalize(cv2.calcHist([im2],[i],None,[256],[0,256]), None, 0, 1, cv2.NORM_MINMAX)
        
        scores.append(cv2.compareHist(hist1, hist2, metric))
    
    return scores

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
    running_sum_chis = [0, 0, 0]
    running_sum_bhats = [0, 0, 0]
    running_sum_inters = [0, 0, 0]
    
    for i in range(len(tube_frames) - 1):
        im1 = tube_frames[i]
        im2 = tube_frames[i+1]
        
        # Compute difference
        chi_scores = compute_similarity(im1, im2, cv2.HISTCMP_CHISQR)
        bhat_scores = compute_similarity(im1, im2, cv2.HISTCMP_BHATTACHARYYA)
        inter_scores = compute_similarity(im1, im2, cv2.HISTCMP_INTERSECT)
        
        #chi_factors = chi_scores / (running_sum_chis / i) if i != 0 else 1
        if i > 0:
            averages = list(map(truediv, running_sum_chis, [i ,i, i]))
            chi_factors = list(map(truediv, chi_scores, averages))
            
            averages = list(map(truediv, running_sum_bhats, [i ,i, i]))
            bhat_factors = list(map(truediv, bhat_scores, averages))
            
            averages = list(map(truediv, running_sum_inters, [i ,i, i]))
            inter_factors = list(map(truediv, inter_scores, averages))
        else:
            chi_factors = [1, 1, 1]
            bhat_factors = [1, 1, 1]
            inter_factors = [1, 1, 1]
        
        # bhat_factors = bhat_scores / (running_sum_bhats / i) if i != 0 else 1
        # inter_factors = inter_scores / (running_sum_inters / i) if i != 0 else 1
        
        running_sum_chis = list(map(add, chi_scores, running_sum_chis))
        running_sum_bhats = list(map(add, bhat_scores, running_sum_bhats))
        running_sum_inters = list(map(add, inter_scores, running_sum_inters))
        
        print('chi FACTORS: ', chi_factors, np.mean(chi_factors))
        print('bhat FACTORS: ', bhat_factors, np.mean(bhat_factors))
        print('inter FACTORS: ', inter_factors, np.mean(inter_factors))
        
        #print('CHI: ', round(chi_score, 2), '|', round(chi_factor, 2))
        # print('BHAT: ', round(bhat_score, 2), '|', round(bhat_factor, 2))
        # print('INTER: ', round(inter_score, 2), '|', round(inter_factor, 2))
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
    
    eval_tube('./results/object_tracking/webServer_inputVideo/140/')
    # for tube_path in sorted(glob(video_path)):
    #     eval_tube(tube_path)


if __name__ == '__main__':
  # Parse user provided arguments
  args = parse_args()
  main(args.video_name)