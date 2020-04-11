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
        "--location", help="path to video folder",
        required = True)
    parser.add_argument(
        "--mode", help="1 = normal, 2 = sort unknowns",
        default = 1)
    return parser.parse_args()

ready = False


def obtain_action(mode):
    if mode == 1:
        actions = {'1':'Handshaking','2':'Hugging','3':'Reading','4':'Drinking','5':'Pushing_Pulling','6':'Carrying','7':'Calling','8':'Running','9':'Walking','10':'Lying','11':'Sitting','12':'Standing', '13':'Unknown'}
    else:
        actions = {'1':'Bin', '2':'Needs_splitting'}

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
                result = result + "_" + actions[a]
            else:
                print(a +" is an invalid action.")
                valid = False
                break
    return result[1:]

def main(location, mode):
    location = 'results/action_recognition/{}'.format(location)

    ground_truths = open('{}/ground_truths.txt'.format(location), 'w')

    for directory in os.listdir(location):
        currentTube = os.path.join(location, directory)
        print("Actiontube: " + currentTube)

        track_id = "_".join(directory.split("_", 2)[:2])

        print("track id = {}".format(track_id))
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
        action = obtain_action(mode)


        output = "{}, {}\n".format(track_id, action)

        print("Writing {}".format(output))
        ground_truths.write(output)

    ground_truths.close()
        

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.location, args.mode)