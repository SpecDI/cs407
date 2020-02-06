import numpy as np
import cv2

def run_test():
    print("test workking please")
    #return

def modify_video(video_name):
    print("Modifing Video....\n")
    #video_name = '..'+ video_name
    video_name= video_name[1:] #Need to remove the first / which is prepended - ie video name is /media/1.1.1.mov - but need media/1.1.1.mov
    print("Attempting to open ", video_name)
    #test = '1.1.1.mov'
    #print("TESTING")

    video = cv2.VideoCapture(video_name)

    if (video.isOpened()== False):
        print("Error opening video stream or file...\n")
    else:
        w = int(video.get(3))
        h = int(video.get(4))
        print(w, h)

        #ALL FOR MP4 etc but dont work
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG') #python3
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V') #python3
        # fourcc = cv2.VideoWriter_fourcc(*'XVID') #python3

        fourcc = cv2.VideoWriter_fourcc(*'vp80') #works - with webm suffix - but extremly slow??? :(

        #fourcc = cv2.cv.CV_FOURCC(*'MJPG') #python


        #out = cv2.VideoWriter('edited_video.avi', fourcc, 20, (w, h)) #.AVI FORM - WORKS BUT CANT DISPLAY
        # out = cv2.VideoWriter('edited_videoMP4.mp5', fourcc, 20.0, (w, h))
        out = cv2.VideoWriter('static/edited_videoWEBMSTATIC.webm', fourcc, 20.0, (w, h))

        frame_number = -1
        while video.isOpened() and (frame_number < 50):
            frame_number+=1
            ret, frame = video.read()
            print(frame_number)
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



#modify_video('/1.1.1.mov')
