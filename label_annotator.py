from __future__ import division, print_function, absolute_import

import os

import cv2
import numpy as np
from PIL import Image

from math import sqrt
from scipy import spatial
import random
import argparse
import time
def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_file", help="Path to file video",
        default = None,
        required=True)
    parser.add_argument(
        "--location", help="Path to file containing bboxes",
        default = None,
        required=True)
    return parser.parse_args()

def getBoxes(frame_number, detections):
    bboxs = []
    for detection in detections:
        test = detection.split()

        if(len(test) > 5):
            if test[5] == str(frame_number):
                bboxs.append([int(test[1]), int(test[2]), int(test[3]), int(test[4])])
        else:
            if test[0] == str(frame_number):
                bboxs.append([int(test[1]), int(test[2]), int(test[3]), int(test[4])])
    return bboxs

def main(sequence_file, location):


    detections_file = "results/object_detection/{}/object_detection.txt".format(location)

    detections = open(detections_file, "r").readlines()

    frame_number = 0
    video_capture = cv2.VideoCapture(sequence_file)

    while video_capture.isOpened():
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        bboxs = getBoxes(frame_number, detections)


        for bbox in bboxs:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.imshow('', cv2.resize(frame, (1200, 675)))
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.sequence_file, args.location)
