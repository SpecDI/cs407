import os
import glob
import json
import shutil
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('./action_recognition/architectures')

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
from keras.applications.vgg16 import preprocess_input

from action_recognition.architectures._5_5_TransferLSTM_TS import TS_CNN_LSTM


from imutils.video import FileVideoStream

object_detection_file = None
object_tracking_directory = None
action_recognition_file = None
batch_number = dict()
current_frame = 0

# Constant variables
FRAME_LENGTH = 80
FRAME_WIDTH = 80
FRAME_NUM = 64

CHANNELS = 3
CLASSES = 13

KERNEL_SHAPE = (3, 3)
POOL_SHAPE = (2, 2)
INPUT_SHAPE = (FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, CHANNELS)

# Action indices
actions_header = sorted(['Unknown', 'Sitting', 'Lying', 'Drinking', 'Calling', 'Reading', 'Handshaking', 'Running', 'Pushing-Pulling', 'Walking', 'Hugging', 'Carrying', 'Standing'])

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

    parser.add_argument(
        "--test_mode", help="Boolean. If true - individual component outputs stored.",
        action = 'store_true',
        default = False
    )

    parser.add_argument(
        "--test_output", help="Output directory for testing information, required if --test_mode=True",
        type = str
    )

    parser.add_argument(
        "--bayesian", help="Boolean. If true - actions will be predicted with uncertainty.",
        action = 'store_true',
        default = False
    )
    
    parser.add_argument(
        "--batch_factor", help="Boolean. If true - batch size = 32, false = 64",
        default = 1
    )

    parser.add_argument(
        "--sourceDir_path", help="Path to the directory containing the input videos. Defaults to web_server",
        required = True
    )
    return parser

def hamming_loss(y_true, y_pred, tval = 0.4):
    tmp = K.abs(y_true - y_pred)
    return K.mean(K.cast(K.greater(tmp, tval), dtype = float))

def process_batch(batch):
    processed_batch = []

    # Pad images
    for img in batch:
        try:
            processed_batch.append(preprocess_input(cv2.resize(img, (FRAME_WIDTH, FRAME_LENGTH))))
        except:
            processed_batch.append(processed_batch[-1])

    return np.asarray(processed_batch)

def predict_with_uncertainty(model, x, n_iter=10):
    """
    Makes predictions using the MC dropout method. 

    param model: Model with dropout layers.
    param x: Input for prediction.

    returns: predictions, uncertainty of prediction
    """

    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    
    result = np.zeros((n_iter, ) + (1, 13))

    for iter in range(n_iter):
        result[iter] = f([x,1])

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)

    return prediction, uncertainty

def interpolateBbox(first, second, ratio):
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
    elif len(currentXs) <=2 or len(currentYs) <= 2:
        return None
    return [min(currentXs), min(currentYs), max(currentXs), max(currentYs)]


def repeatBatch(batch, i):
    a = np.array(batch)

    return np.tile(a, i)

def processFrame(locations, processedFrames, processedTracks, track_tubeMap, track_actionMap, model, test_mode, bayesian, batch_factor):
    global current_frame
    location = locations.pop(0)
    frame = processedFrames.pop(0)
    tracks = processedTracks.pop(0)

    for trackId, bbox in tracks.items():
        # min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
        # cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255),2)
        # cv2.putText(frame, str(trackId),(int(min_x), int(min_y)),0, 5e-3 * 200, (0, 0, 255),2)                
        
        # Init text to be appended to bbox
        append_str = str(trackId)

        if trackId not in track_actionMap:
            track_actionMap[trackId] = ''

        # Init new key if necessary
        if trackId not in track_tubeMap:
            track_tubeMap[trackId] = []

        # Add frame segment to track dict
        block = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].copy()

        if(block.shape[0] == 0 or block.shape[1] == 0):
            if len(track_tubeMap[trackId]) > 0:
                block = track_tubeMap[trackId][-1]
            else:
                continue

        track_tubeMap[trackId].append(block)


        if test_mode:
            track_directory = object_tracking_directory + "/" + str(trackId)
            if not os.path.exists(track_directory):
                os.mkdir(track_directory)
            cv2.imwrite(track_directory + "/" + str(current_frame) + ".jpg", block) 

        # Check size of track bucket


        if len(track_tubeMap[trackId]) == int(FRAME_NUM/batch_factor):
            if test_mode:
                global batch_number
                if trackId in batch_number:
                    batch_number[trackId] += 1
                else:
                    batch_number[trackId] = 0

                # recognition_directory = action_recognition_directory + "/" + str(trackId) + "_" + str(batch_number[trackId])

                # if not os.path.exists(recognition_directory):
                #     os.mkdir(recognition_directory)
                # for i, block in enumerate(track_tubeMap[trackId], current_frame - FRAME_NUM + 1):
                #     cv2.imwrite(recognition_directory + "/" + str(i) + ".jpg", block) 
            # Process action tube
            batch = process_batch(track_tubeMap[trackId])
            
            batch = repeatBatch(batch, batch_factor)


            batch = batch.reshape(1, FRAME_NUM, FRAME_LENGTH, FRAME_WIDTH, 3)

            # Generate predictions
            # Clear the list
            track_tubeMap[trackId] = []

            # Update action label to match corresponding action
            mean_thresh = 3e-1
            max_actions = 3
            results = np.zeros((13, ))

            if bayesian: 
                preds, uncertainty = predict_with_uncertainty(model, batch)
                print("Preds: ", preds)
                print("Uncertainty: ", uncertainty)

                # Threshold by the mean.
                new_uncertainties = np.where(preds > mean_thresh, uncertainty, float("inf"))[0]
                # Get at most three actions with the smallest uncertainty.
                result_ind = np.argsort(new_uncertainties)[:max_actions]
                   
                results[result_ind] = 1
                results = np.where(new_uncertainties == float("inf"), 0., results)
                results2 = [preds[0][x] if results[x] == 1 else results[x] for x, _ in enumerate(results)]
            else:
                preds = model.predict(batch)[0]
                print("Preds: ", preds)
                # Threshold by the mean 
                new_preds = np.where(preds > mean_thresh, preds, float("-inf"))
                # Get at most 3 actions 
                result_ind = np.argsort(new_preds)[-max_actions:]
                results[result_ind] = 1
                results = np.where(new_preds == float("-inf"), 0., results)
                results2 = [preds[x] if results[x] == 1 else results[x] for x, _ in enumerate(results)]

            actions_header_arr = np.array(actions_header)
            action_list = actions_header_arr[results.astype(bool)]
            action_label = ','.join(action_list)



            
            track_identifier = str(trackId) + "_" + str(batch_number[trackId])
            resultString = "[" + ', '.join(map(str, results2)) + "]"
            action_recognition_file.write(track_identifier + " = "+resultString +"\n")

            if not action_label:
                action_label = "Unknown"

            print(f"Person {trackId} is {action_label}")

            # if test_mode:
            #     action_dir_label = '_'.join(action_list)
            #     if not action_dir_label:
            #         action_dir_label = "Unknown"
            #     os.rename(recognition_directory,recognition_directory + "_" + action_dir_label) 

            # Update map
            track_actionMap[trackId] = action_label

        # Update text to be appended
        append_str += ' ' + track_actionMap[trackId]
        # Create bbox and text label

        if not test_mode:
            frame = cv2.rectangle(frame, (int(location[0]), int(location[1])), (int(location[2]), int(location[3])), (0,0, 255), 5)
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            frame = cv2.putText(frame, append_str,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
    current_frame += 1
    return frame


def initialiseTestMode(video_name):
    global object_detection_file
    global object_tracking_directory
    global action_recognition_file

    if not os.path.exists('results/object_detection/' + str(video_name)):
        os.mkdir('results/object_detection/' + str(video_name))
    object_detection_file = open('results/object_detection/' + str(video_name) + '/object_detection.txt', 'w')

    if not os.path.exists('results/object_tracking/' + str(video_name)):
        os.mkdir('results/object_tracking/' + str(video_name))
    object_tracking_directory = 'results/object_tracking/' + str(video_name)

    if not os.path.exists('results/action_recognition/' + str(video_name)):
        os.mkdir('results/action_recognition/' + str(video_name))
    #action_recognition_directory = 'results/action_recognition/' + str(video_name)
    action_recognition_file = open('results/action_recognition/' + str(video_name) + '/action_recogition.txt', 'w')

def validBbox(bbox):
    if abs(bbox[0] - bbox[2])<=1 or abs(bbox[1] - bbox[3]) <=1:
        return False
    return True

def writeFrame(frame, out, hide_window, test_mode):
    if not test_mode:
        out.write(frame)
    if not (hide_window or test_mode):
        frame = cv2.resize(frame, (1200, 675))
        cv2.imshow('', frame)

def main(yolo, hide_window, weights_file, test_mode, test_output, bayesian, batch_factor, input_file):
    if test_mode:
        global object_detection_file
        global object_tracking_directory
        global action_recognition_directory
        initialiseTestMode(test_output)

    print('Starting pipeline...')

    # Define output path based on file name and web_server dir
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f'./web_server/output_{file_name}.avi'

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    model = TS_CNN_LSTM(INPUT_SHAPE, CLASSES)
    model.load_weights('action_recognition/architectures/weights/' + weights_file + '.hdf5')

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

    # Let input stream load some frames
    time.sleep(5)

    # Define the codec and create VideoWriter object
    w = 3840
    h = 2160
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #*'XVID'
    # Build video output handler only if we are not cropping

    out = None
    if not test_mode:
        out = cv2.VideoWriter(output_file, fourcc, 11, (w, h))

    fps = 0.0
    location = (0, 0)


    frame_number = 0

    track_buffer = []

    unprocessedFrames = []

    processedFrames = []

    processedTracks = []

    locations = []

    skip = 1
    while video_capture.more():
        frame = video_capture.read()  # frame shape 640*480*3
        if not isinstance(frame, np.ndarray):
            break
        t1 = time.time()

        x = w
        y = h

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


            features = encoder(frame,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature).rescale(xScale, yScale, diffx, diffy) for bbox, feature in zip(boxs, features)]
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            if test_mode:
                if len(detections) != 0:
                    for i in range(0,len(detections)):
                        bbox = detections[i].to_tlbr()
                        object_detection_file.write(str(frame_number)+' ' + str(int(bbox[0])) + ' '+str(int(bbox[1])) + ' '+str(int(bbox[2])) + ' '+str(int(bbox[3])) + '\n')
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            frame = frameCopy
    
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
                secondTrack = tracks.copy()

                for i, oldFrame in enumerate(unprocessedFrames, 1):
                    tracks = dict()
                    for trackId in secondTrack:
                        if trackId in firstTrack:
                            min_x, min_y, max_x, max_y = interpolateBbox(firstTrack[trackId], secondTrack[trackId], i/skip)
                            result = [x+y for x ,y in zip([min_x, min_y, max_x, max_y], firstTrack[trackId])]
                            tracks[trackId] = result
                
                    processedFrames.append(oldFrame)
                    processedTracks.append(tracks.copy())
                    unprocessedFrames = []
                processedFrames.append(frame)
                processedTracks.append(secondTrack)
                track_buffer.append(secondTrack)

            location = rescale(location, xScale, yScale, 0, 0, True)
            locations.append(location)

            if(frame_number >= skip):
                frame = processFrame(locations, processedFrames, processedTracks, track_tubeMap, track_actionMap, model, test_mode, bayesian, batch_factor)



            currentXs = []
            currentYs = []
            for det in detections:
                bbox = det.to_tlbr()

                currentXs.extend([int(bbox[0]),int(bbox[2])])
                currentYs.extend([int(bbox[1]),int(bbox[3])])

        else:
            unprocessedFrames.append(frame)
            locations.append([0,0,0,0])
            if(frame_number >= skip):
                frame = processFrame(locations, processedFrames, processedTracks, track_tubeMap, track_actionMap, model, test_mode, bayesian, batch_factor)

        # Display video as processed if necessary

            # save a frame

        if(frame_number >= skip):
            writeFrame(frame, out, hide_window, test_mode)


        frame_number += 1
        if frame_number % 5 != 0:
            location = calculateLocation(currentXs, currentYs)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps/skip))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()

    while len(processedFrames) != 0:
        frame_number += 1
        frame = processFrame(locations, processedFrames, processedTracks, track_tubeMap, track_actionMap, model, test_mode, bayesian, batch_factor)
        writeFrame(frame, out, hide_window, test_mode)


    while len(unprocessedFrames) != 0:
        frame_number += 1
        frame = unprocessedFrames.pop()
        writeFrame(frame, out, hide_window, test_mode)
    
    if not test_mode:
        out.release()


    cv2.destroyAllWindows()

    if test_mode:
        object_detection_file.close()

if __name__ == '__main__':
    # Parse user provided arguments
    parser = parse_args()
    args = parser.parse_args()
    if (args.test_mode) and (args.test_output is None):
        parser.error("--test_output required if --test_mode=True")
    
    if FRAME_NUM % int(args.batch_factor) != 0:
        parser.error("--batch_factor must be factor of {}".format(FRAME_NUM))
    # Get file paths
    file_paths = glob.glob(args.sourceDir_path + "*.mp4")
    print(args.sourceDir_path)
    print('\nDiscovered files:')
    for file in file_paths:
        print(file)
        
    for video_path in file_paths:
        print(f'Processing: {video_path}\n')
        main(YOLO(), args.hide_window, args.weights_file, args.test_mode, args.test_output, args.bayesian, int(args.batch_factor), video_path)
