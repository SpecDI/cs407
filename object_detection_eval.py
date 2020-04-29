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

def calculateCentre(bbox):
    x = (bbox[0] + bbox[2])/2

    y = (bbox[1] + bbox[3])/2

    return (x,y)

def closestBbox(bbox, truth_bboxs):
    centre = calculateCentre(bbox)

    truth_centres = list(map(calculateCentre, truth_bboxs))

    minimum = float('inf')

    closest_bbox = None
    for i, truth_centre in enumerate(truth_centres):
        dist = sqrt((centre[0] - truth_centre[0])**2 + (centre[1] - truth_centre[1])**2)
        if dist < minimum:
            closest_bbox = truth_bboxs[i]
            minimum = dist
        


    return closest_bbox

def computeIou(truth_bbox, pred_bbox):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(truth_bbox[0], pred_bbox[0])
    yA = max(truth_bbox[1], pred_bbox[1])
    xB = min(truth_bbox[2], pred_bbox[2])
    yB = min(truth_bbox[3], pred_bbox[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (truth_bbox[2] - truth_bbox[0] + 1) * (truth_bbox[3] - truth_bbox[1] + 1)
    boxBArea = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value

    if iou > 0.1:
        return 1
    return iou

def compareBboxs(frame, test_bboxs, truth_bboxs):
    scores = []
    for truth_bbox in truth_bboxs:
        test_bbox = closestBbox(truth_bbox, test_bboxs)

        if not test_bbox:
            scores.append(0)
            continue
        score = computeIou(test_bbox, truth_bbox)

        scores.append(score)
    return scores


def main(sequence_file, location):


    detections_file = "results/object_detection/{}/object_detection.txt".format(location)
    truths_file = "results/object_detection/{}/ground_truths.txt".format(location)

    detections = open(detections_file, "r").readlines()
    ground_truths = open(truths_file, "r").readlines()

    frame_number = 0
    video_capture = cv2.VideoCapture(sequence_file)


    scores = []
    while video_capture.isOpened():
        print(frame_number)
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        test_bboxs = getBoxes(frame_number, detections)
        truth_bboxs = getBoxes(frame_number, ground_truths)
        

        for bbox in test_bboxs:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        for bbox in truth_bboxs:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        cv2.imshow('', cv2.resize(frame, (1200, 675)))
        scores.extend(compareBboxs(frame, test_bboxs, truth_bboxs))


        
        
        frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

    print(sum(scores) / len(scores))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.sequence_file, args.location)
