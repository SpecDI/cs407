from __future__ import division, print_function, absolute_import

import os

import cv2
import numpy as np
from PIL import Image

from math import sqrt
from scipy import spatial

import argparse

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_file", help="Path to file video",
        default = None,
        required=True)
    parser.add_argument(
        "--detections_file", help="Path to file containing bboxes",
        default = None,
        required=True)
    parser.add_argument(
        "--truths_file", help="Path to file containing truth bboxes",
        default = None,
        required=True)
    return parser.parse_args()

def getBoxes(frame_number, detections):
    bboxs = []
    for detection in detections:
        test = detection.split()
        if test[0] == frame_number:
            bboxs.append([test[1], test[2], test[3], test[4]])
        else:
            break
    return bboxs

def calculateCentre(bbox):
    x = (bbox[0] + bbox[2])/2

    y = (bbox[1] + bbox[3])/2

    return (x,y)

def closestBbox(bbox, truth_bboxs):
    centre = calculateCentre(bbox)

    truth_centres = map(calculateCentre, truth_bboxs)


    minimum = float('inf')

    closest_bbox = None
    for i, truth_centre in enumerate(truth_centres):
        dist = sqrt((centre[0] - truth_centre[0])**2 + (centre[1] - truth_centre[1])**2)
        if dist < minimum:
            closest_bbox = truth_bboxs[i]


    return closest_bbox

def computeMAP(block1, block2):
    return 0

def compareBboxs(frame, test_bboxs, truth_bboxs):
    scores = []
    for test in test_bboxs:
        truth = closestBbox(test, truth_bboxs)

        test_block = frame[int(test[1]):int(test[3]), int(test[0]):int(test[2])].copy()
        truth_block = frame[int(truth[1]):int(truth[3]), int(truth[0]):int(truth[2])].copy()

        score = computeMAP(test_block, truth_block)

        scores.append(score)
    return scores


def main(sequence_file, detections_file, truths_file):


    detections = open(detections_file, "r").readlines()
    ground_truths = open(truths_file, "r").readlines()

    frame_number = 0
    video_capture = cv2.VideoCapture(sequence_file)


    while video_capture.isOpened():
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        test_bboxs = getBoxes(frame_number, detections)
        truth_bboxs = getBoxes(frame_number, ground_truths)

        scores = compareBboxs(frame, test_bboxs, truth_bboxs)

        
        
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.sequence_file, args.detections_file, args.truths_file)
