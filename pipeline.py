import os
import glob
import json
import shutil
import warnings
import sys
warnings.filterwarnings('ignore')

import argparse
from timeit import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from object_detection.yolo3.yolo import YOLO

from tracking.deep_sort import preprocessing
from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tracking.deep_sort.detection import Detection as ddet

from tensorflow.python.keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM

# Constant variables
FRAME_LENGTH = 83
FRAME_WIDTH = 40
FRAME_NUM = 8

# Action indices
actions_header = sorted(['Unknown', 'Sitting', 'Lying', 'Drinking', 'Calling', 'Reading', 'Handshaking', 'Running', 'Pushing_Pulling', 'Walking', 'Hugging', 'Carrying', 'Standing'])

def parse_args():
    """ Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Main detection, tracking and action recognition pipeline")
    return parser.parse_args()

def hamming_loss(y_true, y_pred, tval = 0.4):
    tmp = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.greater(tmp, tval), dtype = float))

def process_batch(batch):
    processed_batch = []

    # Pad images
    for img in batch:
        processed_batch.append(cv2.resize(img, (FRAME_WIDTH, FRAME_LENGTH)))

    return np.asarray(processed_batch)

def cnn_lstm(input_shape, kernel_shape, pool_shape, classes):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=kernel_shape, activation='relu')))

    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=pool_shape)))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(256))
    model.add(Dropout(0.5))

    model.add(Dense(classes, kernel_initializer="normal", name='output'))
    model.add(Activation('softmax'))

    return model

def main(yolo):
    print('Starting pipeline...')

    input_file = './web_server/input.mp4'
    output_file = './web_server/output.avi'

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Load in model
    # model = load_model(
    #     'action_recognition/architectures/weights/lstm.hdf5',
    #     custom_objects={
    #         "hamming_loss": hamming_loss,
    #     }
    # )

    CHANNELS = 3
    CLASSES = 13

    KERNEL_SHAPE = (3, 3)
    POOL_SHAPE = (2, 2)
    INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

    model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)
    model.load_weights('action_recognition/architectures/weights/lstm.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy', hamming_loss])


    # Track id frame batch
    track_tubeMap = {}

    # Track id action
    track_actionMap = {}

    # Image data generator
    datagen = ImageDataGenerator()

    # deep_sort 
    model_filename = 'object_detection/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    video_capture = cv2.VideoCapture(input_file)

    # Define the codec and create VideoWriter object
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #*'XVID'
    # Build video output handler only if we are not cropping
    out = cv2.VideoWriter(output_file, fourcc, 11, (w, h))
    list_file = open('detection.txt', 'w')
    frame_index = -1

    fps = 0.0
    frame_number = 0
    while video_capture.isOpened():
        frame_number+=1
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            
            # Init text to be appended to bbox
            append_str = str(track.track_id)

            if track.track_id not in track_actionMap:
                track_actionMap[track.track_id] = 'Unknown'

            # Init new key if necessary
            if track.track_id not in track_tubeMap:
                track_tubeMap[track.track_id] = []

            # Add frame segment to track dict
            block = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()
            track_tubeMap[track.track_id].append(block / 255.0)

            # Check size of track bucket
            if len(track_tubeMap[track.track_id]) == FRAME_NUM:
                # Process action tube
                batch = process_batch(track_tubeMap[track.track_id])
                batch = batch.reshape(1, FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, 3)
                
                # Generate predictions
                preds = model.predict(batch)[0].tolist()
                print(preds)

                # Clear the list
                track_tubeMap[track.track_id] = []
                # Update action label to match corresponding action
                action_label = actions_header[preds.index(max(preds))]
                print(f"Person {track.track_id} is {action_label}")

                # Update map
                track_actionMap[track.track_id] = action_label

            # Update text to be appended
            append_str += ' ' + track_actionMap[track.track_id]
            # Create bbox and text label
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, append_str,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        cv2.imshow('', cv2.resize(frame, (1200, 675)))

        # save a frame
        out.write(frame)
        frame_index = frame_index + 1
        list_file.write(str(frame_index)+' ')
        if len(boxs) != 0:
            for i in range(0,len(boxs)):
                list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
        list_file.write('\n')


        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(YOLO())
