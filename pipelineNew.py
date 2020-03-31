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

from action_recognition.architectures._1_2_LSTM_OS import cnn_lstm
from action_recognition.architectures.Metrics import MetricsAtTopK
from action_recognition.architectures.Loss import LossFunctions


from imutils.video import FileVideoStream

# Constant variables
FRAME_LENGTH = 200
FRAME_WIDTH = 200
FRAME_NUM = 16

CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

# Action indices
actions_header = sorted(['Unknown', 'Sitting', 'Lying', 'Drinking', 'Calling', 'Reading', 'Handshaking', 'Running', 'Pushing_Pulling', 'Walking', 'Hugging', 'Carrying', 'Standing'])

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Main detection, tracking and action recognition pipeline")

    parser.add_argument(
        "--hide_window", help="Flag used to hide the cv2 output window",
        action = 'store_true',
        default = False
    )

    parser.add_argument(
        "--weights_file", help="Name of weight file to be loaded for the action recognition model",
        required = True)

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

def predict_with_uncertainty(self, model, x, n_iter=10):
    """
    Returns model prediction using MC dropout technique.
    """
    # Implement a function which applies dropout during test time.  
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    
    result = np.zeros((n_iter, ) + (1, 32, 64, 13))

    for iter in range(n_iter):
        result[iter] = f([x,1])

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)

    return prediction, uncertainty

def interpolateBbox(second, first, ratio):
    return (second[0] - first[0]) * ratio, (second[1] - first[1]) * ratio, (second[2] - first[2]) * ratio, (second[3] - first[3]) * ratio

def rescale(bbox, xScale, yScale, diffx, diffy, up):


    if up:
        xStart = (bbox[0] + diffx) * xScale #+ x1
        yStart = (bbox[1] + diffy)  * yScale #+ y1

        xEnd = (bbox[2] + diffx) * xScale #+ x2
        yEnd = (bbox [3] + diffy) * yScale #+ y2
    else:
        xStart = (bbox[0] - diffx) / xScale #+ x1
        yStart = (bbox[1] - diffy)  / yScale #+ y1

        xEnd = (bbox[2] - diffx) / xScale #+ x2
        yEnd = (bbox [3] - diffy) / yScale #+ y2
    return [xStart, yStart, xEnd, yEnd]


def calculateLocation(currentXs, currentYs):
    if not(currentXs) or not(currentYs):
        return None
    return [min(currentXs), min(currentYs), max(currentXs), max(currentYs)]

def processFrame(processedFrames, processedTracks, track_tubeMap, track_actionMap, model):
    frame = processedFrames.pop(0)
    tracks = processedTracks.pop(0)

    for trackId, bbox in tracks.items():
        # min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
        # cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255),2)
        # cv2.putText(frame, str(trackId),(int(min_x), int(min_y)),0, 5e-3 * 200, (0, 0, 255),2)                
        
        # Init text to be appended to bbox
        append_str = str(trackId)

        if trackId not in track_actionMap:
            track_actionMap[trackId] = 'Unknown'

        # Init new key if necessary
        if trackId not in track_tubeMap:
            track_tubeMap[trackId] = []

        # Add frame segment to track dict
        block = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()
        track_tubeMap[trackId].append(block / 255.0)

        # Check size of track bucket
        if len(track_tubeMap[trackId]) == FRAME_NUM:
            # Process action tube
            batch = process_batch(track_tubeMap[trackId])
            batch = batch.reshape(1, FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, 3)
            
            # Generate predictions
            preds = model.predict(batch)[0].tolist()
            print(preds)

            # Clear the list
            track_tubeMap[trackId] = []
            # Update action label to match corresponding action
            action_label = actions_header[preds.index(max(preds))]
            print(f"Person {trackId} is {action_label}")

            # Update map
            track_actionMap[trackId] = action_label

        # Update text to be appended
        append_str += ' ' + track_actionMap[trackId]
        # Create bbox and text label
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, append_str,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
    return frame


def main(yolo, hide_window, weights_file):
    print('Starting pipeline...')

    input_file = './web_server/input.mp4'
    output_file = './web_server/output.avi'

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    metrics = MetricsAtTopK(k=2)
    losses = LossFunctions()

    # Load in model
    # model = load_model(
    #     'action_recognition/architectures/weights/lstm_1_2.hdf5',
    #     custom_objects={
    #         "weighted_binary_crossentropy": losses.weighted_binary_crossentropy,
    #         "recall_at_k": metrics.recall_at_k, 
    #         "precision_at_k": metrics.precision_at_k, 
    #         "f1_at_k": metrics.f1_at_k,
    #         "hamming_loss": hamming_loss,
    #     }
    # )

    model = cnn_lstm(INPUT_SHAPE, KERNEL_SHAPE, POOL_SHAPE, CLASSES)
    model.load_weights('action_recognition/architectures/weights/' + weights_file + '.hdf5')

    metrics = MetricsAtTopK(k=2)
    losses = LossFunctions()
    model.compile(loss=losses.weighted_binary_crossentropy, 
                    optimizer='adam', metrics=['accuracy', 
                                                metrics.recall_at_k, 
                                                metrics.precision_at_k, 
                                                metrics.f1_at_k, 
                                                losses.hamming_loss])

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

    video_capture = FileVideoStream(input_file).start()

    # Define the codec and create VideoWriter object
    w = 3840
    h = 2180
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #*'XVID'
    # Build video output handler only if we are not cropping
    out = cv2.VideoWriter(output_file, fourcc, 11, (w, h))
    list_file = open('detection.txt', 'w')
    frame_index = -1

    fps = 0.0
    location = (0, 0)


    frame_number = 0

    track_buffer = []

    unprocessedFrames = []

    processedFrames = []

    processedTracks = []

    skip = 3
    while video_capture.more():
        frame = video_capture.read()  # frame shape 640*480*3
        if not isinstance(frame, np.ndarray):
            break
        t1 = time.time()

        x = 3840
        y = 2160

        scaledX = 640   
        scaledY = 360

        xScale = x/scaledX
        yScale = y/scaledY


        if(frame_number % skip == 0):
            if (not location) or frame_number % 5 == 0:
                location = [0,0, scaledX, scaledY]
            else:
                location = rescale(location, xScale, yScale, 0, 0, False)

            frameCopy = frame.copy()

            frame = cv2.resize(frame, (scaledX, scaledY), interpolation = cv2.INTER_AREA)
            # image = Image.fromarray(frame)
            image = Image.fromarray(frame[...,::-1]) #bgr to rgb

            diffx = location[0]
            diffy = location[1]

            image = image.crop((location[0], location[1], location[2], location[3]))


            boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
            features = encoder(frame,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature).rescale(xScale, yScale, diffx, diffy) for bbox, feature in zip(boxs, features)]
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            frame = frameCopy
            location = rescale(location, xScale, yScale, 0, 0, True)
            cv2.rectangle(frame, (int(location[0]), int(location[1])), (int(location[2]), int(location[3])), (0,0, 255), 2)
    
            tracks = dict()
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                

                tracks[track.track_id] = bbox



            if frame_number == 0:
                track_buffer.append(tracks)

                processedFrames.append(frame)
                processedTracks.append(tracks)

            else:
                firstTrack = track_buffer.pop(0)
                secondTrack = tracks

                for i, oldFrame in enumerate(unprocessedFrames, 1):
                    tracks = dict()
                    for trackId in secondTrack:
                        if trackId in firstTrack:
                            min_x, min_y, max_x, max_y = interpolateBbox(firstTrack[trackId], secondTrack[trackId], i/skip)
                            result = [x+y for x ,y in zip([min_x, min_y, max_x, max_y], firstTrack[trackId])]
                            tracks[trackId] = result
                    processedFrames.append(oldFrame)
                    processedTracks.append(tracks)
                    unprocessedFrames = []

                processedFrames.append(frame)
                processedTracks.append(secondTrack)
                track_buffer.append(secondTrack)


            if(frame_number > skip - 1):
                frame = processFrame(processedFrames, processedTracks, track_tubeMap, track_actionMap, model)


            currentXs = []
            currentYs = []
            for det in detections:
                bbox = det.to_tlbr()

                currentXs.extend([int(bbox[0]),int(bbox[2])])
                currentYs.extend([int(bbox[1]),int(bbox[3])])

            frame_number += 1
            if frame_number % 5 != 0:
                location = calculateLocation(currentXs, currentYs)
                print(location)

        else:
            unprocessedFrames.append(frame)
            if(frame_number > skip - 1):
                frame = processFrame(processedFrames, processedTracks, track_tubeMap, track_actionMap, model)
            frame_number += 1

        # Display video as processed if necessary
        if frame_number - 1> skip - 1:
            if not hide_window:
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
    main(YOLO(), args.hide_window, args.weights_file)
