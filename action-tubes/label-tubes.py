import cv2
import numpy as np
import os
import shutil
import argparse

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument(
        "--path", help="path to video folder",
        required = True)
    return parser.parse_args()

ready = False


def obtain_action():
    actions = {'1':'Handshaking','2':'Hugging','3':'Reading','4':'Drinking','5':'Pushing_Pulling','6':'Carrying','7':'Calling','8':'Running','9':'Walking','10':'Lying','11':'Sitting','12':'Standing', '13':'Unknown'}

    for k, v in actions.items():
        print(k + ":" +v)

    valid = False
    while not valid:
        valid = True
        action = input("Insert actions, seperated by &: ")
        action = action.replace(" ", "")
        print(action)
        action = action.split("&")
        result = ""
        for a in action:
            if a in actions:
                result = result + "&" + actions[a]
            else:
                print(a +" is an invalid action.")
                valid = False
                break
    return result[1:]

def main(path):
    completed_path = "completed"
    if not os.path.exists(completed_path):
        print("Creating: " + completed_path)
        os.mkdir(completed_path)

    for directory in sorted(os.listdir(path), key=lambda x: int(x.split(".")[0])):
        currentTube = os.path.join(path, directory)
        print("Actiontube: " + currentTube)
        ready = False
        while not ready:
            for filename in sorted(os.listdir(currentTube), key=lambda x: int(x.split(".")[0])) :
                if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    image = cv2.imread(os.path.join(currentTube, filename))
                    cv2.imshow('', image)
                    cv2.waitKey(40)
            cv2.destroyAllWindows()
            val = input("Actiontube complete. Replay? y/[n] ")
            if val != "y":
                ready = True
        action = obtain_action()
        print("Moving " + currentTube + " to " + os.path.join("completed", action, path +"_"+directory))
        
        dest = shutil.move(currentTube, os.path.join("./completed", action, path+"_"+directory))

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.path)