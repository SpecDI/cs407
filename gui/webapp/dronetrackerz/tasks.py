from celery import shared_task, task
from celery_progress.backend import ProgressRecorder
import time
from django.core.files.storage import FileSystemStorage

from time import sleep

import numpy as np
import cv2

import os

@shared_task
def sleepy(duration):
    sleep(duration)
    return None

#@shared_task(bind=True)
@task(bind=True, name="tasks.dummy_modify_video")
def dummy_modify_video(self, vid_name):

    progress_recorder = ProgressRecorder(self)
    print("Progress recorder object :" , progress_recorder)

    print("Modifying Video....\n")
    #video_name = video_name[1:] #Need to remove the first / which is prepended - ie video name is /media/1.1.1.mov - but need media/1.1.1.mov
    print("Attempting to open: " + vid_name)

    print("About to use OpenCV VideoCapture!")
    print("Vid_name data type: ", type(vid_name))
    video = cv2.VideoCapture(str(vid_name))
    print("Just used OpenCV VideoCapture!")

    if (video.isOpened()== False):
        print("Error opening video stream or file...\n")
    else:
        w = int(video.get(3))
        h = int(video.get(4))
        print(w, h)


        BASE_DIR_FOR_DELETION = os.path.dirname(os.path.dirname(__file__))
        fs = FileSystemStorage(os.path.join(BASE_DIR_FOR_DELETION, 'static'))
        print("Deleting previously edited video")
        fs.delete('edited_videoMP4STATIC.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'H264') # WORKING ON WINDOWS :D :D

        out = cv2.VideoWriter('static/edited_videoMP4STATIC.mp4', fourcc, 20.0, (w, h)) # WORKING ON WINDOWS!!!

        frame_number = -1
        frame_limit = 500
        while video.isOpened() and (frame_number < frame_limit):

            frame_number+=1
            ret, frame = video.read()
            print(frame_number)

            progress_recorder.set_progress(frame_number, frame_limit)


            if ret==True:
                frame = cv2.flip(frame,0)
                # write the flipped frame
                out.write(frame)

                #cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        video.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done")
        return None