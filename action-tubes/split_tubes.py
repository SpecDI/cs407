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
    parser.add_argument(
        "--mode", help="1 = normal, 2 = sort unknowns",
        required = True)
    parser.add_argument(
        "--file", help="name of textfile to add unknowns to",
        required=False
    )
    return parser.parse_args()

ready = False


def obtain_action(mode):
    if mode == 1:
        actions = {'1':'Handshaking','2':'Hugging','3':'Reading','4':'Drinking','5':'Pushing_Pulling','6':'Carrying','7':'Calling','8':'Running','9':'Walking','10':'Lying','11':'Sitting','12':'Standing', '13':'Unknown'}
    else:
        actions = {'1':'Bin', '2':'Needs_splitting', "3":"Leave"}

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

def main(path, mode, file):
    completed_path = "completed"
    if not os.path.exists(completed_path):
        print("Creating: " + completed_path)
        os.mkdir(completed_path)
    

    for directory in os.listdir(path):
        if  not directory.startswith("augg"):
            currentTube = os.path.join(path, directory)
            print("Actiontube: " + currentTube)
            ready = False
            while not ready:
                for filename in sorted(os.listdir(currentTube), key=lambda x: int(x.split(".")[0])):
                    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                        image = cv2.imread(os.path.join(currentTube, filename))
                        cv2.imshow('', image)
                        cv2.waitKey(40)
                cv2.destroyAllWindows()
                val = input("Actiontube complete. Replay? y/[n] ")
                if val != "y":
                    ready = True
            action = obtain_action(mode)
            # print("Moving " + currentTube + " to " + os.path.join("completed", action, path +"_"+directory))
            if action == "Needs_splitting":
                print("Adding " + currentTube + " to " + file)
                
                #
                with open(file, "a") as f:
                    f.write(currentTube)
                    f.write("\n")
                
            elif action == "Bin":
                print("moving " + currentTube + " to " + os.path.join("./completed", action, path+"_"+directory))
                shutil.move(currentTube, os.path.join("./completed", action, path+"_"+directory))
            else:
                print("leaving " + currentTube)

if __name__ == '__main__':
    # Parse user provided arguments
    args = parse_args()
    main(args.path, args.mode, args.file)