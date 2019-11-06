import json

def parse_yolo_frame(frame_id, yolo_frame):
    """Convert a dictionary stored in the yolo format to a dictionary stored in mot16 format

    Parameters
    ----------
    frame_id     :  int
        The id of the current frame.
        
    yolo_frame   :  List
        List of dictionary entries for each individual identified in the corresponding frame.

    Returns
    -------
    List
        List of MOT16 based individuals    
    """

    frame_entries = []
    for entity in yolo_frame:
        current_line = str(frame_id) + ' -1 '
        current_line += str(entity['topleft']['x']) + ' '
        current_line += str(entity['topleft']['y']) + ' '

        bb_width = abs(entity['bottomright']['x'] - entity['topleft']['x'])
        current_line += str(bb_width) + ' '

        bb_height = abs(entity['topleft']['y'] - entity['bottomright']['y'])
        current_line += str(bb_height) + ' '
        
        current_line += str(entity['confidence']) + ' -1 -1 -1'

        frame_entries.append(current_line)

    return frame_entries   

def main():
    # Load yolo output from json file
    with open('./yolo_output.json') as json_file:
        yolo_output = json.load(json_file)

    # Iterate through frames and store them in mot like
    # txt file
    with open('./det.txt', 'w') as det_file:
        for i in range(len(yolo_output)):
            mot_sequence = parse_yolo_frame(i + 1, yolo_output[i])
            for entry in mot_sequence:
                det_file.write(entry + '\n')


if __name__ == "__main__":
    main()